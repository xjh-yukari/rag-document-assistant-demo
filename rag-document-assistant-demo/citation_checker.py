import re


def check_citations(
    answer: str,
    source_count: int,
) -> tuple[str, list[str]]:

    warnings = []

    # 找出回答中所有形如 [1]、[2] 的编号
    numbers = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    }

    valid_numbers = {
        number
        for number in numbers
        if 1 <= number <= source_count
    }

    invalid_numbers = numbers - valid_numbers

    if invalid_numbers:
        warnings.append(
            f"回答引用了不存在的来源编号：{sorted(invalid_numbers)}"
        )

        # 将无效引用标记出来，避免误以为它有对应原文
        def replace_invalid(match):
            number = int(match.group(1))

            if 1 <= number <= source_count:
                return match.group(0)

            return "[无效来源]"

        answer = re.sub(
            r"\[(\d+)\]",
            replace_invalid,
            answer,
        )

    if not valid_numbers:
        warnings.append("回答没有有效来源编号，请结合检索原文核对。")

    return answer, warnings