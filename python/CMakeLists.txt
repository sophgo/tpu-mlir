add_subdirectory(transform)
install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  PATTERN "*.py"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
