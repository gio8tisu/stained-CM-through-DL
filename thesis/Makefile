SHELL=/bin/bash
PREFIX=main


.PHONY: all
all : $(PREFIX).acn $(PREFIX).acr $(PREFIX).alg $(PREFIX).aux $(PREFIX).bcf $(PREFIX).bbl $(PREFIX).blg $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc $(PREFIX).tex */*.tex
	pdflatex $(PREFIX)

$(PREFIX).acn $(PREFIX).aux $(PREFIX).bcf $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc : $(PREFIX).tex */*.tex
	pdflatex $(PREFIX)

$(PREFIX).acr $(PREFIX).alg : acronyms.tex
	makeglossaries $(PREFIX)

$(PREFIX).bbl $(PREFIX).blg : refs.bib
	biber $(PREFIX)

.PHONY: clean
clean :  ## Clean output files
	@rm $(PREFIX).{acn,acr,alg,aux,bcf,ist,lof,log,lot,out,run.xml,toc,bbl,blg} # $(PREFIX).glsdefs 
