\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ipsj}[2025/10/31 Simplified IPSJ v4-1 mock]
\LoadClass[a4paper,11pt]{article}

% Basic margin setup similar to IPSJ camera-ready
\RequirePackage{geometry}
\geometry{left=25mm,right=25mm,top=25mm,bottom=25mm}

% Font and section tweaks (lightweight)
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}

% Internal storage for title and author data
\newcommand{\@affiliation}{ }
\newcommand{\@authoremail}{ }
\newcommand{\@keywords}{ }

\renewcommand{\author}[2]{\gdef\@author{#1}\gdef\@authoremail{#2}}
\newcommand{\affiliate}[2]{\gdef\@affiliation{#2}}

\newenvironment{keywords}{\def\keywordname{キーワード}\par\vspace{1ex}\noindent\textbf{\keywordname:} }{\par\vspace{2ex}}

\renewcommand{\maketitle}{%
  \begin{center}
    {\Large\bfseries \@title}\par\vspace{1.5ex}
    {\large \@author}\par\vspace{0.5ex}
    {\normalsize \@affiliation}\par\vspace{0.5ex}
    {\normalsize \@authoremail}\par\vspace{2ex}
  \end{center}
}

% Abstract formatting similar to IPSJ style
\renewenvironment{abstract}{\begin{center}\bfseries 要旨\end{center}\small}{\normalsize\par\vspace{2ex}}

\endinput
