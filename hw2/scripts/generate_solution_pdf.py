from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import fitz


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp" / "pdfs"
OUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUT_DIR / "solution.pdf"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def build_styles():
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=sample["Title"],
            fontName="STSong-Light",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=sample["Heading1"],
            fontName="STSong-Light",
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=sample["Heading2"],
            fontName="STSong-Light",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=sample["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=sample["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
            wordWrap="CJK",
            spaceAfter=6,
        ),
        "mono": ParagraphStyle(
            "MonoBlock",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#111827"),
        ),
    }


STYLES = build_styles()


def p(text, style="body"):
    return Paragraph(text.replace("\n", "<br/>"), STYLES[style])


def formula_block(text, width):
    block = Preformatted(text, STYLES["mono"])
    table = Table([[block]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def sample_indices(frame_count, sample_count):
    if frame_count <= 1:
        return [0]
    if frame_count <= sample_count:
        return list(range(frame_count))
    return sorted({round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)})


def create_contact_sheet(gif_path, output_path, sample_count=4, max_frame_size=(220, 150)):
    with Image.open(gif_path) as gif:
        indices = sample_indices(getattr(gif, "n_frames", 1), sample_count)
        frames = []
        for number, idx in enumerate(indices, start=1):
            gif.seek(idx)
            frame = gif.convert("RGBA")
            frame.thumbnail(max_frame_size, Image.LANCZOS)
            framed = ImageOps.expand(frame, border=2, fill="white")

            canvas = Image.new("RGB", (framed.width, framed.height + 22), "#f8fafc")
            canvas.paste(framed, (0, 0), framed)
            draw = ImageDraw.Draw(canvas)
            draw.text((6, framed.height + 5), f"Frame {number}", fill="#334155")
            frames.append(canvas)

    gap = 12
    sheet_width = sum(frame.width for frame in frames) + gap * (len(frames) - 1) + 24
    sheet_height = max(frame.height for frame in frames) + 24
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rounded_rectangle(
        (0, 0, sheet_width - 1, sheet_height - 1),
        radius=12,
        outline="#cbd5e1",
        width=2,
        fill="white",
    )

    cursor_x = 12
    for frame in frames:
        offset_y = 12 + (sheet_height - 24 - frame.height) // 2
        sheet.paste(frame, (cursor_x, offset_y))
        cursor_x += frame.width + gap

    sheet.save(output_path)
    return output_path


def scaled_image(path, max_width):
    with Image.open(path) as image:
        width, height = image.size
    ratio = max_width / width
    flowable = RLImage(str(path))
    flowable.drawWidth = max_width
    flowable.drawHeight = height * ratio
    return flowable


def draw_page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(doc.pagesize[0] - 18 * mm, 10 * mm, f"{canvas.getPageNumber()}")


def build_story(doc_width):
    animation_dir = ROOT / "animation"
    sheets = {}
    for name in ["uniform", "perpendicular", "attractive", "repulsive", "tangential", "potential_field"]:
        gif_path = animation_dir / f"{name}.gif"
        sheet_path = TMP_DIR / f"{name}_sheet.png"
        sheets[name] = create_contact_sheet(gif_path, sheet_path)

    story = [
        p("Homework 2 \u89e3\u7b54", "title"),
        Spacer(1, 2 * mm),
        p("Question 1", "h1"),
        p(
            "(a) \u6211\u4f1a\u9009\u62e9\u5206\u5c42\u7684\u6df7\u5408\u5f0f\u5bfc\u822a\u67b6\u6784\u3002"
            "\u9ad8\u5c42\u4f7f\u7528\u89c4\u5212\u65b9\u6cd5\u7ed9\u51fa\u5168\u5c40\u76ee\u6807\u548c\u5927\u81f4\u8def\u5f84\uff0c"
            "\u4f4e\u5c42\u4f7f\u7528 behavior-based / reactive \u65b9\u6cd5\u5904\u7406\u5c40\u90e8\u907f\u969c\u3001"
            "\u5b9e\u65f6\u54cd\u5e94\u548c\u6267\u884c\u63a7\u5236\u3002\u8fd9\u6837\u65e2\u4fdd\u7559\u89c4\u5212\u7684\u5168\u5c40\u6027\uff0c"
            "\u53c8\u4fdd\u7559\u53cd\u5e94\u5f0f\u65b9\u6cd5\u7684\u5b9e\u65f6\u6027\u3002"
        ),
        p(
            "(b) \u884c\u4e3a\u5f0f\u8303\u5f0f\u7684\u4e3b\u8981\u4f18\u52bf\u662f\uff1a"
            "\u5b9e\u65f6\u6027\u5f3a\u3001\u6a21\u5757\u5316\u597d\u3001\u5bf9\u73af\u5883\u4e0d\u786e\u5b9a\u6027\u66f4\u9c81\u68d2\u3001"
            "\u5bf9\u7cbe\u786e\u5730\u56fe\u4f9d\u8d56\u8f83\u4f4e\uff0c\u5e76\u4e14\u5bb9\u6613\u5e76\u884c\u7ec4\u5408\u591a\u4e2a\u884c\u4e3a\uff0c"
            "\u4f8b\u5982\u8d8b\u8fd1\u76ee\u6807\u3001\u907f\u969c\u548c\u6cbf\u8fb9\u8fd0\u52a8\u3002"
        ),
        p(
            "(c) \u957f\u671f\u6765\u770b\uff0c\u771f\u6b63\u6709\u7ade\u4e89\u529b\u7684\u662f\u6df7\u5408\u67b6\u6784\u3002"
            "\u89c4\u5212\u8d1f\u8d23\u5168\u5c40\u7ed3\u6784\uff0c\u884c\u4e3a\u5f0f\u65b9\u6cd5\u8d1f\u8d23\u5c40\u90e8\u5b9e\u65f6\u53cd\u5e94\uff0c"
            "\u5b66\u4e60\u65b9\u6cd5\u8d1f\u8d23\u53c2\u6570\u81ea\u9002\u5e94\u4e0e\u7b56\u7565\u6539\u8fdb\u3002"
        ),
        p("Question 2", "h1"),
        p("\u53d8\u91cf\u5b9a\u4e49\uff1a", "h2"),
        formula_block(
            "x = [x, y]^T\n"
            "g = [g_x, g_y]^T\n"
            "o = [o_x, o_y]^T\n"
            "eps = 1e-6\n"
            "max_norm = 2.0",
            doc_width,
        ),
        Spacer(1, 3 * mm),
        p("\u70b9\u5230\u7ebf\u6bb5\u6700\u8fd1\u70b9\uff1a", "h2"),
        formula_block(
            "t = clip(((x - v)^T (w - v)) / ||w - v||^2, 0, 1)\n"
            "p_proj = v + t (w - v)",
            doc_width,
        ),
        Spacer(1, 3 * mm),
        p("\u4e94\u79cd\u539f\u8bed\u529b\u573a\uff1a", "h2"),
        formula_block(
            "Uniform:       F_u = k_u * v / (||v|| + eps)\n"
            "Perpendicular: F_p = k_p * (x - p_proj) / (||x - p_proj|| + eps)\n"
            "Attractive:    F_a = k_a * (g - x), then clip to ||F|| <= 2.0\n"
            "Repulsive:     F_r = k_r * (1/d - 1/d0) / d^2 * (x - o) / (||x - o|| + eps), d < d0\n"
            "Tangential:    F_t = k_t * R_90 * (x - o) / (||x - o|| + eps)\n"
            "R_90 = [[0, -1], [1, 0]]\n"
            "Default params: k_u=1.0, k_p=1.2, k_a=0.6, k_r=1.0, k_t=1.0, d0=2.0",
            doc_width,
        ),
        Spacer(1, 4 * mm),
        p("Q2 \u4eff\u771f\u5173\u952e\u5e27", "h2"),
        p(
            "\u4e0b\u9762 5 \u7ec4\u5173\u952e\u5e27\u5206\u522b\u5bf9\u5e94 uniform\u3001perpendicular\u3001"
            "attractive\u3001repulsive \u548c tangential \u529b\u573a\u3002"
        ),
    ]

    q2_items = [
        ("uniform", "uniform\uff1a\u673a\u5668\u4eba\u6cbf\u56fa\u5b9a\u65b9\u5411\u505a\u8fd1\u4f3c\u5300\u901f\u76f4\u7ebf\u8fd0\u52a8\u3002"),
        ("perpendicular", "perpendicular\uff1a\u673a\u5668\u4eba\u6cbf\u7ebf\u969c\u788d\u6cd5\u5411\u8fdc\u79bb\u969c\u788d\u7269\u3002"),
        ("attractive", "attractive\uff1a\u673a\u5668\u4eba\u671d\u76ee\u6807\u70b9\u6536\u655b\u3002"),
        ("repulsive", "repulsive\uff1a\u673a\u5668\u4eba\u8fdc\u79bb\u7ed9\u5b9a\u70b9\u969c\u788d\u3002"),
        ("tangential", "tangential\uff1a\u673a\u5668\u4eba\u56f4\u7ed5\u70b9\u969c\u788d\u505a\u9006\u65f6\u9488\u7ed5\u884c\u3002"),
    ]

    for key, caption in q2_items:
        story.extend(
            [
                Spacer(1, 2 * mm),
                KeepTogether(
                    [
                        p(caption, "caption"),
                        scaled_image(sheets[key], doc_width),
                        Spacer(1, 3 * mm),
                    ]
                ),
            ]
        )

    story.extend(
        [
            PageBreak(),
            p("Question 3", "h1"),
            p(
                "Q3 \u7684\u76ee\u6807\u662f\u8ba9\u673a\u5668\u4eba\u8d70\u51fa U \u5f62\u969c\u788d\u5185\u90e8\u7684"
                "\u5c40\u90e8\u6781\u5c0f\u503c\u533a\u57df\uff0c\u5e76\u6700\u7ec8\u5230\u8fbe\u771f\u5b9e\u76ee\u6807\u3002"
            ),
            p("\u7b56\u7565\u5206\u4e09\u5c42\uff1a", "h2"),
            p("- \u4fdd\u7559\u76ee\u6807\u5438\u5f15\u529b\uff0c\u589e\u76ca\u53d6 k_a = 0.5\u3002"),
            p(
                "- \u5bf9\u4e09\u6761\u7ebf\u6bb5\u5206\u522b\u4f7f\u7528\u6700\u8fd1\u70b9\u6392\u65a5\u529b\uff0c"
                "\u5e76\u5728\u4e24\u4fa7\u4f7f\u7528\u4e0d\u5bf9\u79f0\u589e\u76ca\uff1a\u5e95\u8fb9 (0.25, 1.2)\uff0c"
                "\u53f3\u8fb9 (1.4, 0.2)\uff0c\u9876\u8fb9 (0.9, 0.3)\uff0c\u7edf\u4e00\u4f5c\u7528\u534a\u5f84 d0 = 1.5\u3002"
            ),
            p(
                "- \u5982\u679c\u8fde\u7eed 15 \u6b65\u5408\u529b\u8303\u6570\u5c0f\u4e8e 0.05\uff0c"
                "\u5219\u8ba4\u4e3a\u6389\u5165\u5c40\u90e8\u9677\u9631\uff0c\u989d\u5916\u53e0\u52a0 10 \u6b65\u5de6\u5411 escape "
                "\u5300\u901f\u573a\uff0c\u5e76\u5c06\u4e34\u65f6\u8131\u56f0\u76ee\u6807\u5207\u6362\u5230 [4.5, 2.7]^T\uff0c"
                "\u76f4\u5230\u673a\u5668\u4eba\u4ece U \u5f62\u5f00\u53e3\u5916\u4fa7\u79bb\u5f00\u540e\u518d\u6062\u590d\u771f\u5b9e\u76ee\u6807\u3002"
            ),
            formula_block(
                "F_att    = 0.5 * (g - x)\n"
                "F_escape = 0.6 * [-1, 0]^T\n"
                "g_escape = [4.5, 2.7]^T\n"
                "Exit condition: x < 4.8 and y < 3.0",
                doc_width,
            ),
            Spacer(1, 4 * mm),
            KeepTogether(
                [
                    p("Q3 \u8131\u56f0\u4eff\u771f\u5173\u952e\u5e27", "h2"),
                    p(
                        "\u5173\u952e\u5e27\u5c55\u793a\u4e86\u673a\u5668\u4eba\u5148\u88ab\u63a8\u51fa U \u5f62\u5185\u90e8\uff0c"
                        "\u518d\u91cd\u65b0\u671d\u771f\u5b9e\u76ee\u6807\u524d\u8fdb\u7684\u8fc7\u7a0b\u3002"
                    ),
                    scaled_image(sheets["potential_field"], doc_width),
                    Spacer(1, 3 * mm),
                ]
            ),
            p(
                "\u7ed3\u679c\uff1a\u5f53\u524d\u5b9e\u73b0\u53ef\u4ee5\u5728\u65e0\u78b0\u649e\u6761\u4ef6\u4e0b\u8d70\u51fa\u5c40\u90e8\u9677\u9631\uff0c"
                "\u5e76\u5230\u8fbe\u76ee\u6807\u70b9\u3002\u9a8c\u8bc1\u65f6\u4eff\u771f\u5728\u7b2c 161 \u6b65\u5230\u8fbe\u76ee\u6807\u3002"
            ),
        ]
    )

    return story


def render_preview(pdf_path):
    document = fitz.open(pdf_path)
    preview_paths = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        output_path = TMP_DIR / f"preview_page_{page_index + 1}.png"
        pix.save(output_path)
        preview_paths.append(output_path)
    return preview_paths


def main():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Homework 2 Solution",
        author="Codex",
    )
    story = build_story(doc.width)
    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    previews = render_preview(PDF_PATH)
    print(PDF_PATH)
    for preview in previews:
        print(preview)


if __name__ == "__main__":
    main()
