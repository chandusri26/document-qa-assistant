import difflib
import re


class QueryCorrector:

    def __init__(
        self,
        metadata,
        similarity_threshold=0.80
    ):
        self.similarity_threshold = similarity_threshold

        self.vocabulary = self._build_vocabulary(
            metadata
        )

    def _build_vocabulary(self, metadata):

        vocabulary = {}

        for item in metadata:

            text = item.get("text", "")

            words = re.findall(
                r"[A-Za-z0-9]+",
                text
            )

            for word in words:

                if len(word) < 4:
                    continue

                key = word.lower()

                if key not in vocabulary:
                    vocabulary[key] = word

        return vocabulary

    def correct_query(self, query):

        words = re.findall(
            r"\b[A-Za-z0-9]+\b",
            query
        )

        corrected_words = []

        for word in words:

            # Keep short words unchanged
            if len(word) < 4:

                corrected_words.append(word)
                continue

            # Keep exact vocabulary matches unchanged
            if word.lower() in self.vocabulary:

                corrected_words.append(word)
                continue

            matches = difflib.get_close_matches(
                word.lower(),
                self.vocabulary.keys(),
                n=1,
                cutoff=self.similarity_threshold
            )

            if matches:

                corrected_word = (
                    self.vocabulary[matches[0]]
                )

                corrected_words.append(
                    corrected_word
                )

            else:

                corrected_words.append(word)

        corrected_query = query

        for original, corrected in zip(
            words,
            corrected_words
        ):

            if original != corrected:

                corrected_query = re.sub(
                    rf"\b{re.escape(original)}\b",
                    corrected,
                    corrected_query,
                    count=1
                )

        return corrected_query