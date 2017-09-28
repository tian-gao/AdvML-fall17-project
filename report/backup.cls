% !TEX encoding = UTF-8 Unicode
%==================================================
%% copyright by Tian Gao
%==================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{layout.cls}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[fancyhdr,fntef,UTF8,adobefonts]{article}

%% Packages
% Basic Packages
\RequirePackage[top=3cm, bottom=2cm, left=2cm, right=2cm, headheight=2cm]{geometry}
\RequirePackage{indentfirst}
\RequirePackage{color}
\RequirePackage{graphicx,psfrag}
\RequirePackage{times}
\RequirePackage{footnote}
\makesavenoteenv{tabular}

% Graphics Packages
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{epsfig}
\RequirePackage{subfigure}
\graphicspath{{figs/}}

% Other Packages
\RequirePackage{fancyhdr}    
\RequirePackage{lastpage}
\RequirePackage{hyperref}
\RequirePackage[compact]{titlesec}
\RequirePackage{lipsum}

%% Margin
\setlength{\parindent}{0em}
\setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}
\linespread{1.2}

%% Figure & Table Captions
\renewcommand\theequation{\arabic{section}--\arabic{equation}}
\renewcommand\thefigure{\arabic{section}--\arabic{figure}}
\renewcommand\thetable{\arabic{section}--\arabic{table}}

%% Header & Footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small{\MakeUppercase{\@title}}}
\fancyhead[R]{\small{\@subtitle}}
\fancyfoot[C]{\thepage}
\setlength{\headsep}{0.5em}

%% Bibliography
\bibliographystyle{unsrt}
% \renewcommand\bibliographyname{References}


%% Definitions
\let\@subtitle\@empty
\let\@authortitle\@empty
\let\@authoremail\@empty
\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\newcommand{\authortitle}[1]{\gdef\@authortitle{#1}}
\newcommand{\authoremail}[1]{\gdef\@authoremail{#1}}

\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}

%% Title Page
\makeatletter
\def\maketitle{%
    \thispagestyle{empty}
    \begin{center}
        \vspace*{1em}
        \LARGE{\bf\@title} \\ [1em]
        \Large{\@subtitle} \\ [1em]

        \begin{minipage}[t]{0.49\textwidth}
        \begin{center} \normalsize
            \textbf{\@author} \\
            \textit{\@authortitle} \\
            \texttt{\@authoremail}
        \end{center}
        \end{minipage} \\ [1em]

        \normalsize{\@date}
        
    \end{center}
    \vspace*{2em}
    % \cleardoublepage
}
\let\ps@plain\ps@fancy
\makeatother