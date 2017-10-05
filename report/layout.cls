% !TEX encoding = UTF-8 Unicode
%==================================================
%% copyright by Tian Gao
%==================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{layout.cls}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[fancyhdr,fntef,UTF8,adobefonts]{book}

%% Packages
% Basic Packages
\RequirePackage[top=3cm, bottom=2cm, left=2cm, right=2cm, headheight=2cm]{geometry}
\RequirePackage{indentfirst}
\RequirePackage{color}
\RequirePackage{graphicx,psfrag}
\RequirePackage{footnote}
\makesavenoteenv{tabular}
\RequirePackage{fontspec}
\newfontfamily\monaco{Monaco}
\setmainfont{Times}

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
\RequirePackage{setspace}

%% Margin
\setlength{\parindent}{0em}
\setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}
\linespread{1.2}

%% Figure & Table Captions
\renewcommand\theequation{\arabic{chapter}--\arabic{equation}}
\renewcommand\thefigure{\arabic{chapter}--\arabic{figure}}
\renewcommand\thetable{\arabic{chapter}--\arabic{table}}

%% Header & Footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small{\@title}}
\fancyhead[R]{\small{\@subtitle}}
\fancyfoot[C]{\thepage}
\setlength{\headsep}{0.5em}

%% Bibliography
\bibliographystyle{unsrt}
\renewcommand\bibname{References}


%% Definitions
\let\@subtitle\@empty
\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}

\titleformat{\chapter}{\Large\bfseries}{\thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{0em}{1em}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}

%% Abstract
\newenvironment{abstract}{\textbf{\large\center Abstract \\ [2em]}}{}

%% Title Page
\makeatletter
\def\maketitle{%
    \thispagestyle{empty}
    \begin{center}
        \vspace*{4em}
        \LARGE{\bf\@title} \\ [2em]
        \Large{\@subtitle} \\ [4em]

        \begin{center} \normalsize
            \textbf{Tian Gao} \\
            \textit{tg2585} \\
            \texttt{tian.gao@columbia.edu} \\ [3em]
        \end{center}
        \begin{center} \normalsize
            \textbf{Tian Gao} \\
            \textit{tg2585} \\
            \texttt{tian.gao@columbia.edu} \\ [3em]
        \end{center}

        \vfill
        \normalsize{GR5242 Advanced Machine Learning Final Project} \\
        \normalsize{\@date}
    \end{center}
    \cleardoublepage
}
\let\ps@plain\ps@fancy
\makeatother


%% code listing
\RequirePackage{listings}
\definecolor{mygreen}{rgb}{0.01,0.5,0.01}
\definecolor{myblue}{rgb}{0.01,0.01,1}
\definecolor{myred}{rgb}{0.73,0.13,0.13}
\definecolor{mycomment}{rgb}{0.25,0.5,0.5}
\lstset{
    tabsize=4,
    escapeinside=``,
    language=python,
    commentstyle=\color{mycomment},
    keywordstyle=\color{mygreen},
    showstringspaces=false,
    stringstyle=\color{myred},
    numbers=left,
    stepnumber=1,
    numberstyle=\tiny,
    basicstyle=\monaco\footnotesize,
    showspaces=false,
    breaklines=true,
}
