# Get the directory of the currently executing script
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Grab all of the argumemts
$arguments = $args -join " "

# Call install.bat with the passed arguments
$process = Start-Process "cmd.exe" -ArgumentList "/c", "$script_dir\install.bat $arguments" -Wait -NoNewWindow -PassThru

# Capture the exit code of the batch script
$exitCode = $process.ExitCode

# Return the exit code
exit $exitCode
