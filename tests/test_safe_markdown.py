import pytest

from safe_markdown import neutralize_links


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("See ![chart](https://evil.example/p.png?d=secret) now", "See chart now"),
        ('Click [here](https://evil.example/claim "title")', "Click here (`https://evil.example/claim`)"),
        ("Visit http://evil.example/claim to verify.", "Visit `http://evil.example/claim` to verify."),
        ("Autolink <https://evil.example>", "Autolink `https://evil.example`"),
        ("Also www.evil.example here", "Also `www.evil.example` here"),
        ("Ref ![a][1] and [b][2]\n[1]: https://evil.example/i.png", "Ref a and b\n"),
        ("Already `https://ok.example` in code", "Already `https://ok.example` in code"),
        ("Article 9(2)(h) and **bold** stay", "Article 9(2)(h) and **bold** stay"),
    ],
)
def test_links_and_images_become_inert(markdown, expected):
    assert neutralize_links(markdown) == expected
