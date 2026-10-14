from resume_assist.ingest import draft_markdown


def test_all_caps_section_title_becomes_a_heading():
    pages = ["JANE DOE\nSUMMARY\nEngineer.\nEXPERIENCE\nAcme Corp"]

    assert draft_markdown(pages) == (
        "# JANE DOE\n\n## SUMMARY\n\nEngineer.\n\n## EXPERIENCE\n\nAcme Corp\n"
    )


def test_wrapped_bullet_is_joined_into_one_list_item():
    pages = [
        "JANE DOE\nEXPERIENCE\n"
        "• Designed an API across design and go-\nlive, supporting \n15 robots.\n"
        "• Mentored five consultants."
    ]

    assert draft_markdown(pages) == (
        "# JANE DOE\n\n## EXPERIENCE\n\n"
        "- Designed an API across design and go-live, supporting 15 robots.\n"
        "- Mentored five consultants.\n"
    )


def test_date_range_ends_the_previous_roles_bullets():
    pages = [
        "JANE DOE\nEXPERIENCE\n"
        "01/2021 - Present\nPrincipal Consultant\n• Led scoping.\n"
        "06/2019 - 01/2021\nSenior Consultant\n• Built integrations."
    ]

    assert draft_markdown(pages) == (
        "# JANE DOE\n\n## EXPERIENCE\n\n"
        "01/2021 - Present\n\nPrincipal Consultant\n\n- Led scoping.\n\n"
        "06/2019 - 01/2021\n\nSenior Consultant\n\n- Built integrations.\n"
    )


def test_page_break_artefacts_are_cleaned_up():
    pages = [
        "JANE DOE\nEXPERIENCE\nWinnersh\xa0Triangle",
        "\xa0/\xa0\nEXPERIENCE\nBengaluru, India",
    ]

    assert draft_markdown(pages) == (
        "# JANE DOE\n\n## EXPERIENCE\n\nWinnersh Triangle\n\nBengaluru, India\n"
    )
