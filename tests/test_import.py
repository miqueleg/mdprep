def test_import_and_version():
    import mdprep

    assert isinstance(mdprep.__version__, str)
    assert mdprep.__version__

