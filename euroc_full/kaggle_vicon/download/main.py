# Kaggle CPU kernel：從 ETH Research Collection（失敗則 robotics.ethz.ch 舊鏡像）只取各序列
# mav0/imu0 與 mav0/state_groundtruth_estimate0，輸出到 /kaggle/working/vicon_room/<序列>/mav0/...
# 只用標準函式庫。以 HTTP Range 直接讀 zip 內的 zip（若外層為 STORED），總流量約 60 MB 而非 12 GB。
import io, os, sys, time, json, zipfile, struct, urllib.request, urllib.error, shutil, csv

OUT = "/kaggle/working/vicon_room"
RC = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/%s/content"
ROOMS = [  # (bitstream uuid, size, 外層資料夾, 序列)
    ("02ecda9a-298f-498b-970c-b7c44334d880", 6042263426, "vicon_room1", ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]),
    ("ea12bc01-3677-4b4c-853d-87c7870b8c44", 6013384949, "vicon_room2", ["V2_01_easy", "V2_02_medium", "V2_03_difficult"]),
]
MIRROR = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/%s/%s/%s.zip"
KEEP = ("/mav0/imu0/", "/mav0/state_groundtruth_estimate0/")
BLK = 2 << 20
MAX_TRIES = 25          # 每個區塊最多重試次數（指數退避，上限 5 分鐘）
UA = {"User-Agent": "Mozilla/5.0"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class HttpFile(io.RawIOBase):
    """唯讀、可 seek 的 HTTP Range 檔案，小型 LRU 區塊快取。"""
    def __init__(s, url, size=None):
        s.url, s.pos, s.cache, s.nreq, s.nbytes = url, 0, {}, 0, 0
        s.size = size if size else s._probe_size()

    def _probe_size(s):
        req = urllib.request.Request(s.url, headers={**UA, "Range": "bytes=0-0"})
        r = urllib.request.urlopen(req, timeout=60)
        cr = r.headers.get("Content-Range")
        if not cr:
            raise IOError("server does not support Range")
        return int(cr.split("/")[1])

    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos

    def seek(s, o, w=0):
        s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o
        return s.pos

    def _block(s, idx):
        if idx in s.cache:
            return s.cache[idx]
        start = idx * BLK
        end = min(start + BLK, s.size) - 1
        for t in range(MAX_TRIES):
            try:
                req = urllib.request.Request(s.url, headers={**UA, "Range": f"bytes={start}-{end}"})
                d = urllib.request.urlopen(req, timeout=60).read()
                s.nreq += 1
                if len(d) != end - start + 1:
                    raise IOError("short read")
                s.nbytes += len(d)
                if len(s.cache) > 16:
                    s.cache.pop(next(iter(s.cache)))
                s.cache[idx] = d
                return d
            except urllib.error.HTTPError as ex:
                wait = min(10 * 2 ** t, 300)
                log(f"  HTTP {ex.code} blk {idx} try {t} wait {wait}s")
                if ex.code in (401, 403, 404):
                    raise
                time.sleep(wait)
            except Exception as ex:
                wait = min(10 * 2 ** t, 300)
                log(f"  {type(ex).__name__}: {ex} blk {idx} try {t} wait {wait}s")
                time.sleep(wait)
        raise IOError(f"giving up block {idx} of {s.url[:60]}")

    def readinto(s, b):
        if s.pos >= s.size:
            return 0
        n = 0
        want = len(b)
        while n < want and s.pos < s.size:
            idx, off = divmod(s.pos, BLK)
            d = s._block(idx)
            k = min(want - n, len(d) - off)
            b[n:n + k] = d[off:off + k]
            n += k
            s.pos += k
        return n


class SubFile(io.RawIOBase):
    """外層 zip 內 STORED 成員的 [off, off+size) 視窗，供內層 ZipFile 使用。"""
    def __init__(s, base, off, size):
        s.base, s.off, s.size, s.pos = base, off, size, 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0):
        s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o
        return s.pos
    def readinto(s, b):
        if s.pos >= s.size:
            return 0
        want = min(len(b), s.size - s.pos)
        s.base.seek(s.off + s.pos)
        d = s.base.read(want)
        b[:len(d)] = d
        s.pos += len(d)
        return len(d)


def extract_from(zf, seq, out_root, prefix=""):
    """從 zf 只解出 mav0/imu0 與 mav0/state_groundtruth_estimate0。"""
    n = 0
    for i in zf.infolist():
        if i.is_dir():
            continue
        name = "/" + i.filename
        if any(k in name for k in KEEP):
            rel = name[name.index("/mav0/") + 1:]
            dst = os.path.join(out_root, seq, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(i) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            n += 1
    return n


def via_research_collection(uid, size, room, seq):
    h = HttpFile(RC % uid, size)
    outer = zipfile.ZipFile(io.BufferedReader(h, buffer_size=1 << 20))
    inner_name = f"{room}/{seq}/{seq}.zip"
    zi = outer.getinfo(inner_name)
    log(f"  inner {inner_name} size={zi.file_size} compress_type={zi.compress_type}")
    if zi.compress_type == zipfile.ZIP_STORED:
        # 讀 local file header 算資料起點
        h.seek(zi.header_offset)
        lh = h.read(30)
        nlen, elen = struct.unpack("<HH", lh[26:30])
        off = zi.header_offset + 30 + nlen + elen
        inner = zipfile.ZipFile(io.BufferedReader(SubFile(h, off, zi.file_size), buffer_size=1 << 20))
        n = extract_from(inner, seq, OUT)
        log(f"  extracted {n} files via range; requests={h.nreq} bytes={h.nbytes}")
        return n
    # 後備：整個內層 zip 串流到暫存再解（約 0.7–1.2 GB）
    tmp = f"/kaggle/temp/{seq}.zip"
    os.makedirs("/kaggle/temp", exist_ok=True)
    with outer.open(inner_name) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    with zipfile.ZipFile(tmp) as iz:
        n = extract_from(iz, seq, OUT)
    os.remove(tmp)
    return n


def via_mirror(room, seq):
    url = MIRROR % (room, seq, seq)
    h = HttpFile(url)
    z = zipfile.ZipFile(io.BufferedReader(h, buffer_size=1 << 20))
    n = extract_from(z, seq, OUT)
    log(f"  mirror extracted {n} files; requests={h.nreq} bytes={h.nbytes}")
    return n


def done(seq):
    b = f"{OUT}/{seq}/mav0"
    return os.path.exists(f"{b}/imu0/data.csv") and os.path.exists(f"{b}/state_groundtruth_estimate0/data.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    src_used = {}
    for uid, size, room, seqs in ROOMS:
        for seq in seqs:
            if done(seq):
                continue
            ok = False
            for name, fn in (("research-collection", lambda: via_research_collection(uid, size, room, seq)),
                             ("robotics.ethz.ch-mirror", lambda: via_mirror(room, seq))):
                log(f"{seq}: trying {name}")
                try:
                    fn()
                    if done(seq):
                        ok = True; src_used[seq] = name; break
                except Exception as ex:
                    log(f"  {name} failed: {type(ex).__name__}: {ex}")
                    shutil.rmtree(f"{OUT}/{seq}", ignore_errors=True)
            if not ok:
                log(f"FAILED {seq}; stop (no hammering)")
                json.dump({"failed": seq, "src": src_used}, open("/kaggle/working/DOWNLOAD_FAILED.json", "w"))
                sys.exit(2)
    report = {"src": src_used, "seqs": {}}
    for _, _, _, seqs in ROOMS:
        for seq in seqs:
            r = {}
            for sub in ("imu0", "state_groundtruth_estimate0"):
                p = f"{OUT}/{seq}/mav0/{sub}/data.csv"
                with open(p) as f:
                    rows = list(csv.reader(f))
                hdr = rows[0]; body = rows[1:]
                r[sub] = {"header": hdr, "ncol": len(hdr), "rows": len(body), "bytes": os.path.getsize(p),
                          "first": body[0], "last": body[-1]}
            report["seqs"][seq] = r
            log(seq, "imu rows", r["imu0"]["rows"], "gt rows", r["state_groundtruth_estimate0"]["rows"],
                "gt ncol", r["state_groundtruth_estimate0"]["ncol"])
    json.dump(report, open("/kaggle/working/download_report.json", "w"), indent=1)
    log("ALL DONE")


main()
