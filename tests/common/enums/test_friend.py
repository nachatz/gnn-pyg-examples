from common.enums.friend import Friend


class TestFriend:
    def test_label_is_title_case(self) -> None:
        assert [friend.label for friend in Friend] == ["Joe", "Nancy", "Bill", "Mei", "Priya", "Omar"]
