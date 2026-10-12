"""Create a 9x6-inner-corner, 20 mm checkerboard on A4 landscape."""

from __future__ import annotations

from pathlib import Path

PAGE_W = 297.0 * 72.0 / 25.4
PAGE_H = 210.0 * 72.0 / 25.4
MM = 72.0 / 25.4
SQUARE_MM = 20.0
INNER_COLS = 9
INNER_ROWS = 6
SQUARE_COLS = INNER_COLS + 1
SQUARE_ROWS = INNER_ROWS + 1
BOARD_W = SQUARE_COLS * SQUARE_MM * MM
BOARD_H = SQUARE_ROWS * SQUARE_MM * MM


def _pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text(commands: list[str], text: str, x: float, y: float, size: float, *, bold: bool = False) -> None:
    font = "/F2" if bold else "/F1"
    commands.append(f"BT {font} {size:.3f} Tf {x:.3f} {y:.3f} Td ({_pdf_text(text)}) Tj ET")


def create_pdf(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    left = (PAGE_W - BOARD_W) / 2.0
    bottom = 35.0 * MM
    commands: list[str] = ["q", "1 1 1 rg", f"0 0 {PAGE_W:.3f} {PAGE_H:.3f} re f", "0 0 0 rg"]

    # Keep all metadata outside the computer-vision target so the board remains
    # a clean 10 x 7 square pattern with exact nominal dimensions.
    _text(commands, "CALIBRATION CHECKERBOARD - PRINT AT 100%", PAGE_W / 2.0 - 169.0, PAGE_H - 40.0, 13.0, bold=True)
    _text(commands, "9 x 6 inner corners | 10 x 7 squares | 20 mm nominal square | overall 200 x 140 mm", PAGE_W / 2.0 - 176.0, PAGE_H - 57.0, 8.5)

    commands.extend(["0 0 0 rg"])

    for row in range(SQUARE_ROWS):
        for col in range(SQUARE_COLS):
            if (row + col) % 2 == 0:
                commands.append("0 0 0 rg")
            else:
                commands.append("1 1 1 rg")
            x = left + col * SQUARE_MM * MM
            y = bottom + (SQUARE_ROWS - 1 - row) * SQUARE_MM * MM
            commands.append(f"{x:.3f} {y:.3f} {SQUARE_MM * MM:.3f} {SQUARE_MM * MM:.3f} re f")

    # A thin outline makes the physical edge easy to inspect without changing
    # the square geometry.
    commands.extend(["0 0 0 RG", "0 0 0 rg", "0.6 w", f"{left:.3f} {bottom:.3f} {BOARD_W:.3f} {BOARD_H:.3f} re S"])

    _text(commands, "After printing: measure the actual square size with a ruler or caliper and enter that measured value.", PAGE_W / 2.0 - 214.0, bottom - 20.0, 8.5)
    _text(commands, "Printer setting: ACTUAL SIZE / 100% (do not use Fit to Page).", PAGE_W / 2.0 - 145.0, bottom - 34.0, 8.5, bold=True)

    # Independent 50 mm check bar for a quick print-scale sanity check.
    bar_x = PAGE_W - 72.0 * MM
    bar_y = 14.0 * MM
    bar_w = 50.0 * MM
    commands.extend([
        "0 0 0 RG", "0 0 0 rg", "0.8 w",
        f"{bar_x:.3f} {bar_y:.3f} m {bar_x + bar_w:.3f} {bar_y:.3f} l S",
        f"{bar_x:.3f} {bar_y - 1.5 * MM:.3f} m {bar_x:.3f} {bar_y + 1.5 * MM:.3f} l S",
        f"{bar_x + bar_w:.3f} {bar_y - 1.5 * MM:.3f} m {bar_x + bar_w:.3f} {bar_y + 1.5 * MM:.3f} l S",
    ])
    _text(commands, "50 mm scale check", bar_x + 20.0, bar_y + 8.5, 7.5)
    commands.append("Q")
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.3f} {PAGE_H:.3f}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    output.write_bytes(data)


if __name__ == "__main__":
    create_pdf(Path("output/pdf/calibration_checkerboard_A4_landscape.pdf"))
