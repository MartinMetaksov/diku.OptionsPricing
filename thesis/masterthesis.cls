%-----------------------------------------------------%
% @author: Tony Beltramelli - www.tonybeltramelli.com %
%       IT University of Copenhagen - www.itu.dk      %
%       creation date: 2015/06/11, version: 0.1       %
%       last update: 2015/12/05, version: 0.2         %
%-----------------------------------------------------%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{masterthesis}[2015/06/11 v0.1 Master's thesis]
\LoadClass[12pt, oneside, letterpaper]{book}

\RequirePackage[english]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{verbatim}
\RequirePackage{subcaption}
\RequirePackage{caption}
\RequirePackage{listings}
\RequirePackage{url}
\RequirePackage{comment}
\RequirePackage{csquotes}
\RequirePackage[toc,page]{appendix}
\RequirePackage[linesnumbered,ruled,commentsnumbered]{algorithm2e}
\usepackage{caption}
\usepackage[table,xcdraw]{xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\usepackage{chngpage}
\usepackage{textpos}
\usepackage{csvsimple}

% Sorting triangles
\DeclareUnicodeCharacter{25B2}{$\blacktriangle$}
\DeclareUnicodeCharacter{25BC}{$\blacktriangledown$}

\RequirePackage{color}
\definecolor{special}{rgb}{0.3, 0.3, 0.3}

\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{\slshape\nouppercase{\leftmark}}
\cfoot{\thepage}

\RequirePackage{hyperref}
\hypersetup{
    colorlinks,
    citecolor=special,
    filecolor=black,
    linkcolor=black,
    urlcolor=special,
}

\expandafter\def\expandafter\normalsize\expandafter{%
    \normalsize
    \setlength\abovedisplayskip{0pt}
    \setlength\belowdisplayskip{3pt}
    \setlength\abovedisplayshortskip{0pt}
    \setlength\belowdisplayshortskip{3pt}
}

\renewcommand\bibname{References}

\RequirePackage{pdfsync}
\synctex=1

\RequirePackage[times]{quotchap}
\definecolor{chaptergrey}{rgb}{0.5,0.5,0.5}

\RequirePackage{setspace} 
\onehalfspacing

\RequirePackage{multirow}

\parindent 12pt
\parskip 1em

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\newcommand{\abstractpage}{
	\newpage
	\chapter*{Abstract}
	\input{frontmatter/abstract}
	\addcontentsline{toc}{chapter}{\numberline{}Abstract}
}

\newcommand{\acknowledgments}{
    \newpage
	\chapter*{Acknowledgments}
	\input{frontmatter/acknowledgments}
	\addcontentsline{toc}{chapter}{\numberline{}Acknowledgments}
}

\newcommand{\contents}{
    \newpage
    \addcontentsline{toc}{chapter}{\numberline{}Contents}
    \setlength{\parskip}{0pt}
    \setcounter{tocdepth}
    \tableofcontents
}

% \newcommand{\listfigures}{
%     \newpage
%     \addcontentsline{toc}{chapter}{\numberline{}List of figures}
%     \listoffigures
% }

% \newcommand{\listtables}{
%     \newpage
%     \addcontentsline{toc}{chapter}{\numberline{}List of tables}
%     \setlength{\parskip}{1em}
%     \listoftables
% }

\newcommand{\chapterheadline}[1]{
    \renewcommand\chapterheadstartvskip{\vspace*{#1\baselineskip}}
}

\newcommand{\makebibliography}{
    \newpage
    \chapterheadline{-4}
    
    \addcontentsline{toc}{chapter}{\numberline{}Bibliography}
    \bibliography{references}
    \bibliographystyle{plain}
    
    \chapterheadline{-8}
}

\newcommand{\makefrontmatter}{
    \setcounter{page}{1}
    \pagenumbering{roman}
    
    \chapterheadline{-4}
    
    \abstractpage
    \acknowledgments
    \contents
    \listfigures
    \listtables
    
    \clearpage
    
    \setcounter{page}{1}
    \pagenumbering{arabic}
    
    \chapterheadline{-8}
}

\newcommand{\source}[1]{\caption*{Source: {#1}} }

\def\degree#1{\gdef\@degree{#1}}
\def\university#1{\gdef\@university{#1}}
\def\location#1{\gdef\@location{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}

\renewcommand{\maketitle}{
    \begin{titlepage}
        \begin{center}
            \setlength{\parskip}{0pt}
            
            \begin{center}
                \includegraphics[width=7cm]{img/ku_logo}
            \end{center}
            
            \vspace*{10mm}
            
            {\Large {Master Thesis} \par}
            \vspace*{5mm}
            
            {\huge {\bfseries \@title} \par}
            
            \vspace*{10mm}
            
            {\large {\textit{Authors:} \@author} \par}
                        
            \vspace*{5mm}

            {\large {\textit{Supervisors:} \@supervisor} \par}
            
            \vspace*{10mm}
            
            {\large \textsc{\@university} \\ \@location \par}
            
            \vspace*{5mm}
            
            {\large \@date \par}
        \end{center}
    \end{titlepage}
}

% Appendix chapter
\newcommand{\appchapter}[1]{%
\newgeometry{top=3cm}
  \begingroup
    \let\@makechapterhead\@gobble
    \pagestyle{fancy}
    
    \chapter{#1}
    \begin{textblock*}{0cm}(0,-1cm)
        \textbf{\Large{Appendix \Alph{chapter}. #1}}
    \end{textblock*}
  
  \endgroup
}
