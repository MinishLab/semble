import os
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from semble.types import ContentType

MAX_FILE_BYTES = int(os.environ.get("SEMBLE_MAX_FILE_BYTES", 1_000_000))  # Max file size to read and index
_EMPTY_FILE_BYTES = 128
# Languages reachable from more than one file extension. Naming each one here
# keeps the extension table below free of repeated language literals.
_LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "abl":              (".cls", ".p", ".w"),
    "ada":              (".ada", ".adb", ".ads"),
    "asciidoc":         (".adoc", ".asciidoc"),
    "asm":              (".asm", ".s"),
    "bash":             (".bash", ".sh"),
    "batch":            (".bat", ".cmd"),
    "bitbake":          (".bb", ".bbappend", ".bbclass"),
    "c":                (".c", ".h"),
    "c3":               (".c3", ".c3i", ".c3t"),
    "clojure":          (".clj", ".cljc", ".cljs"),
    "cobol":            (".cbl", ".cob", ".cobol"),
    "commonlisp":       (".cl", ".lisp"),
    "cpp":              (".cc", ".cpp", ".cxx", ".hpp", ".hxx"),
    "cuda":             (".cu", ".cuda"),
    "devicetree":       (".dts", ".dtsi"),
    "diff":             (".diff", ".patch"),
    "dot":              (".dot", ".gv"),
    "eex":              (".eex", ".leex"),
    "elixir":           (".ex", ".exs"),
    "erlang":           (".erl", ".hrl"),
    "forth":            (".4th", ".fth"),
    "fortran":          (".f", ".f03", ".f08", ".f90", ".f95"),
    "fsharp":           (".fs", ".fsx"),
    "gap":              (".g", ".gi"),
    "gn":               (".gn", ".gni"),
    "gnuplot":          (".gnuplot", ".gp", ".plt"),
    "godot_resource":   (".tres", ".tscn"),
    "graphql":          (".gql", ".graphql"),
    "groovy":           (".gradle", ".groovy"),
    "html":             (".htm", ".html"),
    "ini":              (".cfg", ".ini"),
    "javascript":       (".cjs", ".js", ".jsx", ".mjs"),
    "jinja2":           (".j2", ".jinja2"),
    "jsonnet":          (".jsonnet", ".libsonnet"),
    "kotlin":           (".kt", ".kts"),
    "ledger":           (".journal", ".ldg", ".ledger"),
    "make":             (".makefile", ".mk"),
    "markdown":         (".markdown", ".md"),
    "mermaid":          (".mermaid", ".mmd"),
    "netlinx":          (".axi", ".axs"),
    "nginx":            (".conf", ".nginx"),
    "nim":              (".nim", ".nims"),
    "perl":             (".pl", ".pm"),
    "po":               (".po", ".pot"),
    "postscript":       (".eps", ".ps"),
    "powershell":       (".ps1", ".psd1", ".psm1"),
    "python":           (".py", ".pyi", ".pyw"),
    "razor":            (".cshtml", ".razor"),
    "rescript":         (".res", ".resi"),
    "sml":              (".fun", ".sig", ".sml"),
    "sourcepawn":       (".inc", ".sp"),
    "squirrel":         (".nut", ".squirrel"),
    "starlark":         (".bzl", ".star"),
    "systemverilog":    (".sv", ".svh"),
    "terraform":        (".tf", ".tfvars"),
    "textproto":        (".pbtxt", ".textproto"),
    "typescript":       (".cts", ".mts", ".ts"),
    "typoscript":       (".tsconfig", ".typoscript"),
    "vhdl":             (".vhd", ".vhdl"),
    "xml":              (".xml", ".xsl", ".xslt"),
    "yaml":             (".yaml", ".yml"),
}

# Extensions whose language is unique to them.
_SINGLE_EXTENSION_LANGUAGE = {
    ".agda":               "agda",
    ".al":                 "al",
    ".as":                 "actionscript",
    ".astro":              "astro",
    ".awk":                "awk",
    ".beancount":          "beancount",
    ".bib":                "bibtex",
    ".bicep":              "bicep",
    ".blade":              "blade",
    ".bq":                 "sql_bigquery",
    ".brs":                "brightscript",
    ".bsl":                "bsl",
    ".caddyfile":          "caddy",
    ".cairo":              "cairo",
    ".capnp":              "capnp",
    ".cedar":              "cedar",
    ".cedarschema":        "cedarschema",
    ".cel":                "cel",
    ".cfc":                "cfml",
    ".chatito":            "chatito",
    ".circom":             "circom",
    ".ck":                 "chuck",
    ".clar":               "clarity",
    ".cmake":              "cmake",
    ".cook":               "cooklang",
    ".corn":               "corn",
    ".cpon":               "cpon",
    ".cr":                 "crystal",
    ".cs":                 "csharp",
    ".css":                "css",
    ".cst":                "cst",
    ".csv":                "csv",
    ".cue":                "cue",
    ".cylc":               "cylc",
    ".d":                  "d",
    ".dart":               "dart",
    ".desktop":            "desktop",
    ".dhall":              "dhall",
    ".dj":                 "djot",
    ".dl":                 "souffle",
    ".dockerfile":         "dockerfile",
    ".dsp":                "faust",
    ".dtd":                "dtd",
    ".ebnf":               "ebnf",
    ".eds":                "eds",
    ".el":                 "elisp",
    ".elm":                "elm",
    ".elv":                "elvish",
    ".enforce":            "enforce",
    ".erb":                "embeddedtemplate",
    ".fc":                 "func",
    ".fidl":               "fidl",
    ".filter":             "poe_filter",
    ".fir":                "firrtl",
    ".fish":               "fish",
    ".fnl":                "fennel",
    ".fsd":                "facility",
    ".fsi":                "fsharp_signature",
    ".gd":                 "gdscript",
    ".gdshader":           "gdshader",
    ".gitattributes":      "gitattributes",
    ".gitignore":          "gitignore",
    ".gleam":              "gleam",
    ".glsl":               "glsl",
    ".go":                 "go",
    ".gotmpl":             "gotmpl",
    ".gren":               "gren",
    ".hack":               "hack",
    ".hare":               "hare",
    ".hbs":                "glimmer",
    ".hcl":                "hcl",
    ".heex":               "heex",
    ".hjson":              "hjson",
    ".hlsl":               "hlsl",
    ".hocon":              "hocon",
    ".hoon":               "hoon",
    ".hs":                 "haskell",
    ".http":               "http",
    ".hurl":               "hurl",
    ".hx":                 "haxe",
    ".idr":                "idris",
    ".ino":                "arduino",
    ".ispc":               "ispc",
    ".jai":                "jai",
    ".janet":              "janet",
    ".java":               "java",
    ".jl":                 "julia",
    ".jq":                 "jq",
    ".json":               "json",
    ".json5":              "json5",
    ".just":               "just",
    ".k":                  "kcl",
    ".kdl":                "kdl",
    ".lc":                 "elsa",
    ".lds":                "linkerscript",
    ".lean":               "lean",
    ".less":               "less",
    ".liquid":             "liquid",
    ".ll":                 "llvm",
    ".lua":                "lua",
    ".luau":               "luau",
    ".m":                  "objc",
    ".magik":              "magik",
    ".matlab":             "matlab",
    ".meson":              "meson",
    ".ml":                 "ocaml",
    ".mli":                "ocaml_interface",
    ".mlir":               "mlir",
    ".mll":                "ocamllex",
    ".mod":                "gomod",
    ".mojo":               "mojo",
    ".move":               "move",
    ".nasm":               "nasm",
    ".ncl":                "nickel",
    ".ninja":              "ninja",
    ".nix":                "nix",
    ".norg":               "norg",
    ".nqc":                "nqc",
    ".nu":                 "nushell",
    ".odin":               "odin",
    ".org":                "org",
    ".pas":                "pascal",
    ".pem":                "pem",
    ".pgn":                "pgn",
    ".php":                "php",
    ".pkl":                "pkl",
    ".pony":               "pony",
    ".pp":                 "puppet",
    ".prisma":             "prisma",
    ".pro":                "prolog",
    ".promql":             "promql",
    ".properties":         "properties",
    ".proto":              "proto",
    ".prql":               "prql",
    ".psv":                "psv",
    ".pug":                "pug",
    ".purs":               "purescript",
    ".ql":                 "ql",
    ".qml":                "qmljs",
    ".r":                  "r",
    ".rasi":               "rasi",
    ".rb":                 "ruby",
    ".rbs":                "rbs",
    ".re":                 "re2c",
    ".rego":               "rego",
    ".rkt":                "racket",
    ".robot":              "robot",
    ".roc":                "roc",
    ".ron":                "ron",
    ".rs":                 "rust",
    ".rst":                "rst",
    ".rtf":                "rtf",
    ".scad":               "openscad",
    ".scala":              "scala",
    ".scm":                "scheme",
    ".scss":               "scss",
    ".shtml":              "superhtml",
    ".slang":              "slang",
    ".smali":              "smali",
    ".smithy":             "smithy",
    ".smk":                "snakemake",
    ".sol":                "solidity",
    ".sparql":             "sparql",
    ".sql":                "sql",
    ".st":                 "smalltalk",
    ".stan":               "stan",
    ".svelte":             "svelte",
    ".sw":                 "sway",
    ".swift":              "swift",
    ".tact":               "tact",
    ".tal":                "uxntal",
    ".tape":               "vhs",
    ".tcl":                "tcl",
    ".td":                 "tablegen",
    ".templ":              "templ",
    ".tera":               "tera",
    ".tex":                "latex",
    ".thrift":             "thrift",
    ".tl":                 "teal",
    ".tla":                "tlaplus",
    ".todotxt":            "todotxt",
    ".toml":               "toml",
    ".trigger":            "apex",
    ".tsp":                "typespec",
    ".tsv":                "tsv",
    ".tsx":                "tsx",
    ".ttl":                "turtle",
    ".twig":               "twig",
    ".typst":              "typst",
    ".v":                  "v",
    ".vb":                 "vb",
    ".verilog":            "verilog",
    ".vim":                "vim",
    ".vrl":                "vrl",
    ".vue":                "vue",
    ".wast":               "wast",
    ".wat":                "wat",
    ".wgsl":               "wgsl",
    ".wit":                "wit",
    ".wl":                 "wolfram",
    ".yuck":               "yuck",
    ".zig":                "zig",
    ".ziggy":              "ziggy",
    ".zsh":                "zsh",
}

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    **_SINGLE_EXTENSION_LANGUAGE,
    **{
        extension: language
        for language, extensions in _LANGUAGE_EXTENSIONS.items()
        for extension in extensions
    },
}


_DOC_LANGUAGES = {
    "asciidoc",
    "bibtex",
    "djot",
    "doxygen",
    "html",
    "javadoc",
    "jsdoc",
    "latex",
    "luadoc",
    "markdown",
    "markdown_inline",
    "mermaid",
    "norg",
    "norg_meta",
    "org",
    "phpdoc",
    "po",
    "rst",
    "rtf",
    "vimdoc",
}

_CONFIG_LANGUAGES = {
    "beancount",
    "capnp",
    "cedarschema",
    "comment",
    "cooklang",
    "cpon",
    "desktop",
    "devicetree",
    "diff",
    "dtd",
    "editorconfig",
    "ebnf",
    "git_config",
    "gitattributes",
    "gitcommit",
    "gitignore",
    "godot_resource",
    "gomod",
    "gosum",
    "gowork",
    "gpg",
    "hjson",
    "hocon",
    "ini",
    "kdl",
    "ledger",
    "pem",
    "pgn",
    "properties",
    "proto",
    "requirements",
    "ron",
    "smithy",
    "ssh_config",
    "textproto",
    "thrift",
    "todotxt",
    "toml",
    "turtle",
    "typespec",
    "wit",
    "xcompose",
    "xml",
    "yaml",
    "ziggy_schema",
}

_DATA_LANGUAGES = {
    "csv",
    "json",
    "json5",
    "psv",
    "tsv",
}


ALL_LANGUAGES = frozenset(_EXTENSION_TO_LANGUAGE.values())
_CODE_LANGUAGES = ALL_LANGUAGES - _DOC_LANGUAGES - _CONFIG_LANGUAGES - _DATA_LANGUAGES
_LANGUAGE_TO_EXTENSIONS: defaultdict[str, list[str]] = defaultdict(list)
for _extension, _language in _EXTENSION_TO_LANGUAGE.items():
    _LANGUAGE_TO_EXTENSIONS[_language].append(_extension)

_CONTENT_TYPE_LANGUAGES = {
    ContentType.CODE: _CODE_LANGUAGES,
    ContentType.DOCS: _DOC_LANGUAGES,
    ContentType.CONFIG: _CONFIG_LANGUAGES,
}


def detect_language(file_name: Path) -> str | None:
    """Detects the language of a file."""
    return _EXTENSION_TO_LANGUAGE.get(file_name.suffix.lower())


def get_extensions(types: Sequence[ContentType]) -> list[str]:
    """Returns a list of supported file extensions for the given content types."""
    return sorted(
        {
            ext
            for content_type in types
            for lang in _CONTENT_TYPE_LANGUAGES[content_type]
            for ext in _LANGUAGE_TO_EXTENSIONS.get(lang, [])
        }
    )


class FileStatus(str, Enum):
    NEWER = "newer"
    TOO_LARGE = "too_large"
    EMPTY = "empty"
    VALID = "valid"


def read_file_text(file_path: Path) -> str:
    """Read a file's text content, replacing invalid UTF-8 characters."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def get_file_status(file_path: Path, write_time: float | None) -> FileStatus:
    """Checks if a file should be indexed based on its size and modification time."""
    stat = file_path.stat()
    if write_time is not None and stat.st_mtime > write_time:
        # Index invalid, file invalid
        return FileStatus.NEWER
    size = stat.st_size
    if size > MAX_FILE_BYTES:
        # index valid, file invalid
        return FileStatus.TOO_LARGE
    if size < _EMPTY_FILE_BYTES and not read_file_text(file_path).strip():
        # index valid, file invalid
        return FileStatus.EMPTY

    # Both valid
    return FileStatus.VALID
