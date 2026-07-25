"""HTML/PDF export of the personalized portfolio proposal.

PDF conversion relies on the external ``wkhtmltopdf`` binary. When it is not
installed the caller can still offer the HTML report; the app degrades
gracefully instead of crashing.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.optimization.black_litterman import PortfolioPerformance

#: Common install locations checked in addition to the system PATH.
_WKHTMLTOPDF_CANDIDATES = (
    Path("C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"),
    Path("/usr/local/bin/wkhtmltopdf"),
    Path("/usr/bin/wkhtmltopdf"),
)


def find_wkhtmltopdf() -> str | None:
    """Locate the ``wkhtmltopdf`` executable, or return ``None``."""
    on_path = shutil.which("wkhtmltopdf")
    if on_path:
        return on_path
    for candidate in _WKHTMLTOPDF_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def render_report_html(
    user_name: str,
    portfolio: pd.DataFrame,
    esg_weights: dict[str, float],
    performance: PortfolioPerformance,
    chart_png_base64: str | None = None,
) -> str:
    """Render the portfolio proposal as a standalone HTML document.

    Args:
        user_name: Display name used in the report headline.
        portfolio: Recommendation table with ``종목명``/``제안 비중``/``E``/
            ``S``/``G``/``종목 소개`` columns.
        esg_weights: The user's pillar weights (slider values).
        performance: Annualized portfolio metrics.
        chart_png_base64: Optional base64-encoded PNG of the allocation chart.

    Returns:
        Complete HTML document as a string.
    """
    chart_html = (
        f'<img src="data:image/png;base64,{chart_png_base64}" '
        'alt="ESG 포트폴리오 파이차트" class="img">'
        if chart_png_base64 else ""
    )

    rows_html = ""
    for _, row in portfolio.sort_values(by="제안 비중", ascending=False).iterrows():
        rows_html += f"""<tr>
            <td>{row['종목명']}</td>
            <td>{row['제안 비중']:.2f}%</td>
            <td>{int(row['E'])}</td>
            <td>{int(row['S'])}</td>
            <td>{int(row['G'])}</td>
            <td style="text-align: left;">{row['종목 소개']}</td>
            </tr>
            """

    return f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ESG 포트폴리오 제안서</title>
        <style>
            body {{ text-align: center; font-family: Pretendard, sans-serif; }}
            .block {{ display: table; width: 100%; margin: 20px auto; }}
            .box {{ display: table-cell; vertical-align: middle; padding: 10px; }}
            .img {{ width: 100%; max-width: 300px; }}
            table {{ margin: auto; }}
            th, td {{ text-align: center; padding: 10px; border: 1px solid #ddd; }}
            th {{ font-size: 15px; background-color: #e3edfa; }}
            .detail-table-container {{ width: 100%; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h1 style="color: #666666;">{user_name}을 위한 ESG 중심 포트폴리오 제안서</h1>
        <p>다음은 {user_name}의 ESG 선호도를 바탕으로 최적화된 포트폴리오 비중입니다.</p>
        <div class="block">
            <div class="box">{chart_html}</div>
            <div class="box">
                <br>
                <h2 style="font-size:20px;">ESG 관심도</h2>
                <table style="width: 90%;">
                    <tr><th>환경</th><td>{esg_weights['environmental']}</td></tr>
                    <tr><th>사회</th><td>{esg_weights['social']}</td></tr>
                    <tr><th>거버넌스</th><td>{esg_weights['governance']}</td></tr>
                </table>
                <h2 style="font-size:20px;">포트폴리오 정보</h2>
                <table style="width: 90%;">
                    <tr><th>예상 수익률</th><td>{performance.expected_return:.2%}</td></tr>
                    <tr><th>예상 변동성</th><td>{performance.volatility:.2%}</td></tr>
                    <tr><th>샤프 비율</th><td>{performance.sharpe_ratio:.2f}</td></tr>
                </table>
            </div>
        </div>
        <div class="detail-table-container">
            <table class="detail-table">
                <thead>
                <tr>
                    <th rowspan='2'>종목</th>
                    <th rowspan='2'>제안 비중</th>
                    <th colspan="3">ESG Score<br>(2023)</th>
                    <th rowspan='2'>종목 소개</th>
                </tr>
                <tr><th>E</th><th>S</th><th>G</th></tr>
                </thead>
                <tbody>
                {rows_html}
                </tbody>
                <tfoot>
                <tr>
                    <td colspan="6" style="font-size:15px; text-align: left;">
                        <p>해당 차트의 환경(E), 사회(S), 거버넌스(G)의 점수는 2023년 기준 점수입니다.</p>
                    </td>
                </tr>
                </tfoot>
            </table>
        </div>
    </body>
    </html>
    """


def html_to_pdf(html_content: str) -> bytes | None:
    """Convert an HTML report to PDF bytes via ``wkhtmltopdf``.

    Returns:
        PDF bytes, or ``None`` when ``wkhtmltopdf`` is not installed.
    """
    import pdfkit

    executable = find_wkhtmltopdf()
    if executable is None:
        return None

    config = pdfkit.configuration(wkhtmltopdf=executable)
    options = {
        "enable-local-file-access": None,
        "encoding": "UTF-8",
        "no-pdf-compression": "",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "report.html"
        pdf_path = Path(tmp_dir) / "report.pdf"
        html_path.write_text(html_content, encoding="utf-8")
        pdfkit.from_file(str(html_path), str(pdf_path),
                         configuration=config, options=options)
        return pdf_path.read_bytes()
