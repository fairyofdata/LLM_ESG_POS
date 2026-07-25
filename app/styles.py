"""Shared UI styling helpers (Pretendard font, hover tooltips, form CSS)."""

import streamlit as st

#: Loads the Pretendard webfont and applies it globally.
GLOBAL_FONT_CSS = """
    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: Pretendard;
        }
    </style>
    """

#: Centers radio button groups (used by the survey and result pages).
RADIO_CENTER_CSS = (
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>'
    '<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>'
)


def inject_global_font() -> None:
    """Apply the Pretendard font to the whole page."""
    st.markdown(GLOBAL_FONT_CSS, unsafe_allow_html=True)


def inject_centered_radios() -> None:
    """Center-align radio widgets."""
    st.write(RADIO_CENTER_CSS, unsafe_allow_html=True)


def page_header(text: str) -> None:
    """Render a bold page banner in the Streamlit header area."""
    st.markdown(
        f"""
        <style>
            header[data-testid="stHeader"]::after {{
                content: "\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0{text}";
                display: block;
                font-size: 30px;
                word-spacing: 3px;
                font-weight: bold;
                color: #999999;
                padding: 10px;
                font-family: Pretendard;
            }}
            a {{ font-family: Pretendard; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_text_on_hover(hover_text: str, index: int, origin_text: str) -> None:
    """Render ``origin_text`` with a tooltip revealed on hover.

    Args:
        hover_text: HTML content of the tooltip.
        index: Suffix making the generated CSS classes unique per widget.
        origin_text: The always-visible label.
    """
    hover_class = f"hoverable_{index}"
    tooltip_class = f"tooltip_{index}"
    text_popup_class = f"text-popup_{index}"

    hover_css = f"""
        .{hover_class} {{
            position: relative;
            display: block;
            cursor: pointer;
            text-align: center;
            font-family: Pretendard;
        }}
        .{hover_class} .{tooltip_class} {{ display: none; }}
        .{hover_class}:hover .{tooltip_class} {{ opacity: 1; }}
        .{text_popup_class} {{
            display: none;
            position: absolute;
            background-color: #f1f1f1;
            padding: 8px;
            border-radius: 4px;
            width: 80%;
            left: 50%;
            transform: translateX(-50%);
            max-width: 200px;
            font-family: Pretendard;
            color: #333;
            font-size: 14px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .{hover_class}:hover .{text_popup_class} {{
            display: block;
            z-index: 999;
        }}
    """
    text_hover = f"""
        <div class="{hover_class}">
            <a href="#hover_text" style="color: #999999; font-family: Pretendard; font-size: 20px; text-align: center; text-decoration: none;font-weight:bold;">{origin_text}&ensp;&ensp;</a>
            <div class="{tooltip_class}"></div>
            <div class="{text_popup_class}">{hover_text}</div>
        </div>
    """
    st.markdown(f"<p>{text_hover}<style>{hover_css}</style></p>", unsafe_allow_html=True)
