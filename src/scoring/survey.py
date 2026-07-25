"""ESG preference survey: questions, scoring matrix, and derived metrics.

Each of the 15 survey questions maps onto one ESG pillar (E/S/G) and onto the
rating agencies whose methodology covers that topic. Scoring a response adds
agency-specific weights to the pillar row, producing a 3x5 preference matrix
(pillars x agencies) that later personalizes the Black-Litterman views.

The weight values reproduce the scoring table designed for the graduation
project (per-agency item coverage derived from the agencies' published
methodologies; see the final report in ``docs/``).
"""

from dataclasses import dataclass

import pandas as pd

#: Answer options offered for every preference question.
CARE_LEVEL_OPTIONS: tuple[str, ...] = ("신경 쓴다.", "보통이다.", "신경 쓰지 않는다.")

#: Numeric response level per answer.
CARE_LEVEL_SCORES: dict[str, float] = {
    "신경 쓴다.": 1.0,
    "보통이다.": 0.5,
    "신경 쓰지 않는다.": 0.0,
}

#: Investment style options (question 16).
INVESTMENT_STYLES: tuple[str, ...] = (
    "ESG 요소를 중심적으로 고려한다.",
    "ESG와 재무적인 요소를 모두 고려한다.",
    "재무적인 요소를 중심적으로 고려한다.",
)

#: Black-Litterman tau per investment style. In the posterior
#: ``mu = inv(tau*Sigma + P'inv(Omega)P) (tau*Sigma*mu_mkt + P'inv(Omega)P*Q)``
#: a LARGER tau weights market-implied returns more heavily, so the
#: financial-centered style gets the largest tau and the ESG-centered style
#: the smallest.
STYLE_TAU: dict[str, float] = {
    "재무적인 요소를 중심적으로 고려한다.": 2.0,
    "ESG와 재무적인 요소를 모두 고려한다.": 1.0,
    "ESG 요소를 중심적으로 고려한다.": 0.5,
}

PILLARS: tuple[str, ...] = ("E", "S", "G")
AGENCY_COLUMNS: tuple[str, ...] = ("esg1", "sandp", "sustain", "iss", "msci")


@dataclass(frozen=True)
class SurveyQuestion:
    """A single preference question and its scoring contribution.

    Attributes:
        text: Question shown to the user (Korean).
        pillar: ESG pillar the question belongs to (``"E"``/``"S"``/``"G"``).
        weights: Per-agency ``(full, half)`` contributions added to the
            pillar row when the user answers "신경 쓴다." (full) or
            "보통이다." (half). "신경 쓰지 않는다." contributes nothing.
    """

    text: str
    pillar: str
    weights: dict[str, tuple[float, float]]


#: The 15 preference questions in display order.
SURVEY_QUESTIONS: tuple[SurveyQuestion, ...] = (
    SurveyQuestion(
        "1. 투자할 때 기업이 탄소 배출이나 오염물질 관리 등 자연을 보호하는 데 신경 쓰는지 고려하시나요?",
        "E", {"sustain": (1.0, 0.25), "msci": (0.5, 0.125)}),
    SurveyQuestion(
        "2. 투자할 때 기업이 환경 관리 시스템을 구축하는 등 기후 변화에 적극 대응하는지 고려하시나요?",
        "E", {"iss": (0.33, 0.0825), "sandp": (1.0, 0.25)}),
    SurveyQuestion(
        "3. 투자할 때 기업이 생산 과정에서 친환경적으로 제품과 서비스를 제공하는지 고려하시나요?",
        "E", {"iss": (0.33, 0.0825), "esg1": (1.0, 0.25)}),
    SurveyQuestion(
        "4. 투자할 때 기업이 자원을 효율적으로 사용하고 배출량을 줄이는지 고려 하시나요?",
        "E", {"iss": (0.33, 0.0825)}),
    SurveyQuestion(
        "5. 투자할 때 기업이 신재생에너지를 활용하는 등 친환경적으로 활동하는지  고려하시나요?",
        "E", {"msci": (0.5, 0.125)}),
    SurveyQuestion(
        "6. 투자할 때 기업이 직원의 안전을 보장하고 소비자의 권리를 보호하는지 고려하시나요?",
        "S", {"sustain": (0.25, 0.0625), "msci": (0.2, 0.05)}),
    SurveyQuestion(
        "7. 투자할 때 기업이 지역사회와의 관계를 잘 유지하고 공정하게 운영하는지 고려하시나요?",
        "S", {"sustain": (0.25, 0.0625), "msci": (0.2, 0.05), "iss": (0.33, 0.0825)}),
    SurveyQuestion(
        "8. 투자할 때 기업이 건강과 사회에 미치는 부정적인 영향을 줄이는지 고려하시나요?",
        "S", {"msci": (0.2, 0.05)}),
    SurveyQuestion(
        "9. 투자할 때 기업이 직원에게 차별 없이 워라벨을 지켜주고, 역량 개발을 지원하는지 고려하시나요?",
        "S", {"iss": (0.33, 0.0825), "esg1": (1.0, 0.25)}),
    SurveyQuestion(
        "10. 투자할 때 기업이 환경 보호, 직원 복지, 공정 거래 등 사회적 책임을 다하는지 고려하시나요?",
        "S", {"sustain": (0.25, 0.0625), "iss": (0.33, 0.0825)}),
    SurveyQuestion(
        "11. 투자할 때 기업이 개인정보 보호 등 사이버 보안을 잘 관리하는지 고려하시나요?",
        "S", {"sustain": (0.25, 0.0625), "msci": (0.2, 0.05), "sandp": (1.0, 0.25)}),
    SurveyQuestion(
        "12. 투자할 때 기업이 경영 구조를 유지하기 위해 이사회의 독립성과 전문성을 높이려는 것을 고려하시나요?",
        "G", {"sustain": (0.25, 0.25), "msci": (0.2, 0.25), "iss": (0.2, 0.0825),
              "sandp": (1.0, 0.0825), "esg1": (0.2, 0.0825)}),
    SurveyQuestion(
        "13. 투자할 때 기업이 감사팀을 운영하고 회계 규정을 잘 지키는지 고려하시나요?",
        "G", {"iss": (0.33, 0.0825), "sandp": (0.33, 0.0825), "esg1": (0.33, 0.0825)}),
    SurveyQuestion(
        "14. 투자할 때 기업이 주주의 권리를 보호하고 이익을 돌려주는지 고려하시나요?",
        "G", {"iss": (0.33, 0.0825), "esg1": (0.33, 0.0825)}),
    SurveyQuestion(
        "15. 투자할 때 기업이 나라에 미치는 영향을 잘 관리하고, 새로운 경영 방식을 도입하는 것을 고려하시나요?",
        "G", {"sandp": (0.33, 0.165), "esg1": (0.33, 0.165)}),
)

INVESTMENT_STYLE_QUESTION: str = "16. 귀하는 투자시 무엇을 고려하시나요?"


def score_survey(answers: list[str]) -> pd.DataFrame:
    """Convert the 15 survey answers into the pillar-by-agency score matrix.

    Args:
        answers: One answer (a :data:`CARE_LEVEL_OPTIONS` value) per question,
            in the same order as :data:`SURVEY_QUESTIONS`.

    Returns:
        Frame indexed by ``E``/``S``/``G`` with one column per agency.

    Raises:
        ValueError: If the number of answers does not match the questions.
    """
    if len(answers) != len(SURVEY_QUESTIONS):
        raise ValueError(
            f"expected {len(SURVEY_QUESTIONS)} answers, got {len(answers)}")

    result = pd.DataFrame(0.0, index=list(PILLARS), columns=list(AGENCY_COLUMNS))
    for question, answer in zip(SURVEY_QUESTIONS, answers):
        level = CARE_LEVEL_SCORES.get(answer, 0.0)
        if level == 0.0:
            continue
        slot = 0 if level == 1.0 else 1
        for agency, weights in question.weights.items():
            result.at[question.pillar, agency] += weights[slot]
    return result


def max_pillar_scores() -> dict[str, float]:
    """Return the maximum attainable score sum per pillar.

    Used to normalize a user's pillar scores into the ``[0, 1]`` slider range.
    """
    maxima = {pillar: 0.0 for pillar in PILLARS}
    for question in SURVEY_QUESTIONS:
        maxima[question.pillar] += sum(full for full, _ in question.weights.values())
    return maxima


def normalized_pillar_preferences(survey_result: pd.DataFrame) -> dict[str, float]:
    """Normalize the pillar score sums to ``[0, 1]`` slider defaults.

    Args:
        survey_result: Matrix produced by :func:`score_survey`.

    Returns:
        Mapping with keys ``environmental``/``social``/``governance``.
    """
    maxima = max_pillar_scores()
    return {
        "environmental": float(survey_result.loc["E"].sum()) / maxima["E"],
        "social": float(survey_result.loc["S"].sum()) / maxima["S"],
        "governance": float(survey_result.loc["G"].sum()) / maxima["G"],
    }


def esg_interest_percent(investment_style: str) -> float:
    """Estimate the user's ESG interest share (in percent) from question 16."""
    style_factor = 0.5 if investment_style == "재무적인 요소를 중심적으로 고려한다." else 1.0
    return 1.0 / (style_factor + 2.0) * 100


def tau_for_style(investment_style: str) -> float:
    """Return the Black-Litterman ``tau`` matching an investment style."""
    return STYLE_TAU.get(investment_style, 1.0)
