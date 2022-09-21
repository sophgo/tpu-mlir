# You can set these variables from the command line.

# module_name === project in conf.py, use \_ instead of _ if module name contains _
module_name := TPU-MLIR
output_file_prefix := tpu-mlir_quick_start

SOURCEDIR_en     = source_en
pdf_name_en      = $(output_file_prefix)_en.pdf

SOURCEDIR_zh    = source_zh
pdf_name_zh      = $(output_file_prefix)_zh.pdf

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR_en      = build_en
BUILDDIR_zh      = build_zh


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR_en)" "$(BUILDDIR_en)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR_zh)" "$(BUILDDIR_zh)" $(SPHINXOPTS) $(O)

pdf: latex
	@cd $(BUILDDIR_en)/latex && xelatex $(module_name).tex
	@cd $(BUILDDIR_en)/latex && xelatex $(module_name).tex
	@mv $(BUILDDIR_en)/latex/$(module_name).pdf $(BUILDDIR_en)/"${pdf_name_en}" && rm -rf $(BUILDDIR_en)/latex

	@cd $(BUILDDIR_zh)/latex && xelatex $(module_name).tex
	@cd $(BUILDDIR_zh)/latex && xelatex $(module_name).tex
	@mv $(BUILDDIR_zh)/latex/$(module_name).pdf $(BUILDDIR_zh)/"${pdf_name_zh}" && rm -rf $(BUILDDIR_zh)/latex

web: html
	#@python3 -m http.server --directory build/html

clean:
	@rm -rf $(BUILDDIR_en)
	@rm -rf $(BUILDDIR_zh)
