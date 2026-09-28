import json
import socket
import struct
import numpy as np


def recvall(conn: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = conn.recv(n - got)
        if not chunk:
            raise ConnectionError(f"socket closed while receiving {n} bytes, got {got}")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send_packet(conn: socket.socket, meta: dict, arrays: dict[str, np.ndarray] | None = None) -> int:
    """
    Send:
      8-byte header length
      JSON header
      raw array payloads

    Return actual sent bytes, including the 8-byte header and JSON metadata.
    """
    arrays = arrays or {}

    array_infos = []
    payloads = []

    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes(order="C")
        array_infos.append({
            "name": name,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "nbytes": len(raw),
        })
        payloads.append(raw)

    header = dict(meta)
    header["arrays"] = array_infos

    header_raw = json.dumps(header, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack("!Q", len(header_raw)))
    conn.sendall(header_raw)

    total = 8 + len(header_raw)
    for raw in payloads:
        conn.sendall(raw)
        total += len(raw)

    return total


def recv_packet(conn: socket.socket) -> tuple[dict, dict[str, np.ndarray], int]:
    """
    Receive one packet.

    Return:
      meta, arrays, actual_recv_bytes
    """
    raw_len = recvall(conn, 8)
    header_len = struct.unpack("!Q", raw_len)[0]
    header_raw = recvall(conn, header_len)
    meta = json.loads(header_raw.decode("utf-8"))

    arrays = {}
    total = 8 + header_len

    for info in meta.get("arrays", []):
        nbytes = int(info["nbytes"])
        raw = recvall(conn, nbytes)
        total += nbytes

        dtype = np.dtype(info["dtype"])
        shape = tuple(info["shape"])
        arr = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        arrays[info["name"]] = arr

    return meta, arrays, total