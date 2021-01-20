import morfeusz2
import string


class Parser:

    def __init__(self):
        self.morf = morfeusz2.Morfeusz()
        stopwords_file = open('sources/stopwords.txt', 'r', encoding="UTF-8")
        self.stopwords = stopwords_file.read().splitlines()

    def remove_punctuation(self, sentence):
        result = ""
        for i in sentence:
            if i not in string.punctuation:
                result += i
        return result

    def remove_stop_words(self, words):
        new_words = set()
        for w in words:
            if w.lower() not in self.stopwords:
                new_words.add(w)
        return new_words

    def lemmatise(self, sentence):
        analysis = self.morf.analyse(sentence)
        first_occurence_for_index = 0
        result = set()
        for i, j, (orth, base, tag, posp, kwal) in analysis:
            if i == first_occurence_for_index:
                result.add(base.split(':')[0])
                first_occurence_for_index += 1
        return result

    def prepare_sentence(self, sentence):
        sentence = self.remove_punctuation(sentence)
        words = self.lemmatise(sentence)
        words = self.remove_stop_words(words)
        return words


