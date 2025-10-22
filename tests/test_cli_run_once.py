import pytest

from casc_lite.cli.run_once import render_prompt


def test_render_prompt_with_placeholder():
    template = "Solve this: {question}"
    prompt = render_prompt(" 1+1= ? ", template)
    assert prompt == "Solve this: 1+1= ?"


def test_render_prompt_without_placeholder():
    template = "Solve the following problem."
    prompt = render_prompt("Find 2+2.", template)
    assert prompt == "Solve the following problem.\n\nFind 2+2."


def test_render_prompt_none_template():
    prompt = render_prompt("   What is 5?   ", None)
    assert prompt == "What is 5?"
