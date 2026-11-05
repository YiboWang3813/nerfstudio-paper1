# TMI template maintained with this manuscript

Active venue/stage: TMI/draft (pre-submission).

The template and paper are maintained together under manuscript/. This
template/ directory groups supporting class/style, bibliography and logo
files. ../main.tex contains the document class, packages, metadata and section
composition and is the only file directly in manuscript/.
Template files are normal versioned files and may be modified for this paper.
Use Git branches for venue and submission-stage variants.

Imported from Journals_Conferences_Templates, commit
8c31f1cd11521382a508e4e0e85768e817ecb22f,
venues/tmi/2023: ieeecolor.cls, generic.sty and logo.pdf.
IEEEtran.bst and IEEEabrv.bib were copied from the TeX Live installation
used by the previous build. Original copyright/license notices in these files
remain intact. The repository no longer depends on a template submodule.

The imported class already contains the compatibility changes listed below.
These are inherited changes, not a claim of a newly downloaded TMI template.

# Reusable local patches

`ieeecolor.cls` is based on the original file in
`archives/IEEE_TMI_Template_Latex.zip` and includes the following reusable
fixes recovered from the paper repository's Git history:

1. Replace the legacy tabular-based running heads with a vbox implementation
   that avoids alignment errors in current LaTeX distributions.
2. Correct the malformed bibliography glue expression
   `\itemsep 0pt plus pt` to `\itemsep 0pt plus 0pt`.
3. Load `xcolor` directly instead of `color` to avoid incompatible repeated
   color-definition warnings in projects that already use `xcolor`.
4. Increase `\headheight` from `6pt` to `10pt` to accommodate the running
   head and rule without header-height warnings.
5. Restore the visible bibliography heading by changing the empty colored
   section title to `\refname`.
6. Remove the hard-coded `.eps` suffix from logo inclusion and provide a
   preconverted `logo.pdf`, allowing pdfLaTeX to build without writing a
   generated conversion file into the template dependency directory.

These changes are template-level compatibility fixes. Paper-specific spacing,
running-head text, bibliography database paths, and section composition are
intentionally excluded.
