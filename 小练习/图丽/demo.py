import difflib

def _fuzzy_matching(self, plate):
        for i in self._plates:
            if difflib.SequenceMatcher(None, i, plate).quick_ratio() > 0.70:  # 相似度
                return False
        return True
