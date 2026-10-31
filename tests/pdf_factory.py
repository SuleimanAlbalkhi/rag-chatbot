"""Write minimal text PDFs for tests, without a PDF library."""

from pathlib import Path


def make_pdf(path: Path, pages: list[str], page_labels: str | None = None) -> Path:
    """One PDF page per string. `page_labels` is an optional raw /PageLabels dictionary,
    e.g. '<< /Nums [0 << /S /r >> 2 << /S /D >>] >>' (roman i, ii, then 1, 2, ...)."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objs: list[bytes | None] = [None, None, font]
    kids = []
    for text in pages:
        lines = [text[i : i + 90] for i in range(0, len(text), 90)] or [""]
        ops = " ".join(f"({line.replace('(', '').replace(')', '')}) Tj T*" for line in lines)
        stream = f"BT /F1 10 Tf 50 750 Td 14 TL {ops} ET".encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objs)
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        kids.append(len(objs))
    labels = f" /PageLabels {page_labels}" if page_labels else ""
    objs[0] = f"<< /Type /Catalog /Pages 2 0 R{labels} >>".encode()
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()

    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
    return path
