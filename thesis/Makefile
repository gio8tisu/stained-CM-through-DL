PREFIX=main

$(PREFIX).pdf : $(PREFIX).acn $(PREFIX).aux $(PREFIX).bcf $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc $(PREFIX).acr $(PREFIX).alg $(PREFIX).bbl $(PREFIX).blg
	pdflatex $(PREFIX)

$(PREFIX).acn $(PREFIX).aux $(PREFIX).bcf $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc : $(PREFIX).tex
	pdflatex $(PREFIX)

$(PREFIX).acr $(PREFIX).alg :
	makeglossaries $(PREFIX)

$(PREFIX).bbl $(PREFIX).blg :
	biber $(PREFIX)

.PHONY: clean
clean :  ## Clean output files
	rm $(PREFIX).acn $(PREFIX).acr $(PREFIX).alg $(PREFIX).aux $(PREFIX).bcf $(PREFIX).glsdefs $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc $(PREFIX).bbl $(PREFIX).blg
