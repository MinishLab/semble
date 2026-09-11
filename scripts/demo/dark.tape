Output "assets/images/demo-dark.gif"

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 672
Set Theme "Catppuccin Mocha"
Set Padding 20
Set Margin 30
Set MarginFill "#11111b"
Set BorderRadius 12
Set WindowBar Colorful
Set WindowBarSize 36
Set TypingSpeed 45ms

Hide
Type "exec env -i HOME=$DEMO_HOME PATH=$SEMBLE_BIN_DIR:/usr/bin:/bin TERM=xterm-256color /bin/bash --norc --noprofile"
Enter
Sleep 800ms
Type "cd $HOME; PS1='$ '; clear"
Enter
Sleep 300ms
Show

Type "semble install"
Enter
Sleep 1200ms
Enter
Sleep 500ms
Enter
Sleep 500ms
Enter
Sleep 2200ms

Hide
Type "clear"
Enter
Sleep 300ms
Show

Type "cd pydantic"
Enter
Sleep 600ms

Type@25ms `semble search "Where is snake_case converted to camelCase?" -k 1 --format text`
Enter
Wait@60s /\$\s*$/
Sleep 3500ms
