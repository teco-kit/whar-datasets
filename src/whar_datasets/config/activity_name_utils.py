from typing import Iterable, List


def canonicalize_activity_name_list(activity_names: Iterable[object]) -> List[str]:
    canonical: List[str] = []
    for name in activity_names:
        text = str(name).strip()
        for sep in ("-", "_", "/", "(", ")", ",", ":", ";"):
            text = text.replace(sep, " ")

        split_parts: List[str] = []
        for raw_part in text.split():
            current = ""
            for char in raw_part:
                if current and char.isupper() and current[-1].islower():
                    split_parts.append(current)
                    current = char
                else:
                    current += char
            if current:
                split_parts.append(current)

        canonical.append(
            " ".join(
                part if part.isdigit() else part.capitalize() for part in split_parts
            )
        )

    return canonical
