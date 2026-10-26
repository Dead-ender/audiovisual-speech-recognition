"""Character-level CTC text encoding."""

DEFAULT_CHARSET = " abcdefghijklmnopqrstuvwxyz'"


class TextCodec:
    """Maps lowercase transcripts to CTC labels; zero is reserved for blank."""

    def __init__(self, charset: str = DEFAULT_CHARSET) -> None:
        if len(set(charset)) != len(charset):
            raise ValueError("charset characters must be unique")
        self.charset = charset
        self.char_to_id = {char: index + 1 for index, char in enumerate(charset)}
        self.id_to_char = {index: char for char, index in self.char_to_id.items()}
        self.blank_id = 0

    def normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def encode(self, text: str) -> list[int]:
        normalized = self.normalize(text)
        unsupported = sorted(set(normalized) - set(self.char_to_id))
        if unsupported:
            chars = ", ".join(repr(char) for char in unsupported)
            raise ValueError(f"transcript contains unsupported characters: {chars}")
        return [self.char_to_id[char] for char in normalized]

    def decode(self, labels: list[int] | tuple[int, ...]) -> str:
        try:
            return "".join(self.id_to_char[int(label)] for label in labels)
        except KeyError as error:
            raise ValueError(f"unknown character label: {error.args[0]}") from error
