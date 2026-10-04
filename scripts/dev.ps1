#Requires -Version 5.1
<#
.SYNOPSIS
    Spins the whole TokenMiner stack up on this machine.

.DESCRIPTION
    Brings up, in order: PostgreSQL (Docker), the database schema, the .NET API, and the Stratum
    proxy. Values come from .env, which is created from .env.example on first run.

    By default the API and the proxy run natively, which keeps their logs readable and their
    process ids trackable. Pass -Containers to run the proxy and nginx from docker compose
    instead (the API always runs natively so it can be debugged).

    Two optional pieces of the developer loop are opt-in. Pass -MockPool to also start the mock
    Stratum pool from mock-pool/ (Pearl on 3335, Quantus on 3336), and -Portal to also start the
    management-portal Vite dev server on 5273.

    Pass -Debug to raise every component's log level: the proxy runs with RUST_LOG=debug (which
    is where the verbatim Stratum relay is logged, both directions), the API with Serilog at
    Debug, and the mock pool at debug.

.EXAMPLE
    ./scripts/dev.ps1
    ./scripts/dev.ps1 up
    ./scripts/dev.ps1 status
    ./scripts/dev.ps1 logs -Component api -Follow
    ./scripts/dev.ps1 restart -Service api
    ./scripts/dev.ps1 down
    ./scripts/dev.ps1 help

.EXAMPLE
    # Proxy behind nginx in Docker; API stays on the host.
    ./scripts/dev.ps1 up -Containers

.EXAMPLE
    # Everything, including the mock pool and the management portal.
    ./scripts/dev.ps1 up -MockPool -Portal

.EXAMPLE
    # The whole stack with debug logging (proxy relay, API, mock pool).
    ./scripts/dev.ps1 up -MockPool -Portal -Debug

.EXAMPLE
    # Tear everything down and delete the database volume.
    ./scripts/dev.ps1 down -Purge
#>
param(
    # The first positional argument. Deliberately no [Parameter()] attribute: that alone makes the
    # script an advanced function, which reserves -Debug and rejects the switch below.
    [ValidateSet('up', 'down', 'restart', 'status', 'logs', 'migrate', 'help')]
    [string]$Command = 'up',

    # Run the stratum proxy (and nginx) from docker compose rather than natively.
    [switch]$Containers,

    # Also start the mock Stratum pool from mock-pool/ (Pearl on 3335, Quantus on 3336).
    [switch]$MockPool,

    # Also start the management-portal Vite dev server (port 5273).
    [switch]$Portal,

    # With 'down': also delete the PostgreSQL volume, destroying all local data.
    [switch]$Purge,

    # With 'restart': restart a single service instead of the whole stack.
    [ValidateSet('all', 'postgres', 'api', 'proxy', 'mock-pool', 'portal')]
    [string]$Service = 'all',

    # Which component 'logs' should read.
    [ValidateSet('api', 'proxy', 'mock-pool', 'portal', 'all')]
    [string]$Component = 'all',

    # With 'logs': keep reading as new lines arrive.
    [switch]$Follow,

    <#
        Raise every component's log level to debug: proxy RUST_LOG=debug, API Serilog Debug, mock
        pool debug. Declared here rather than relying on the common -Debug parameter so the level
        can be applied to the child processes.

        Note: this is why the script has no [CmdletBinding()] - it would reserve -Debug and reject
        this parameter. Unknown arguments are rejected explicitly below instead.
    #>
    [switch]$Debug,

    # With 'logs': how many existing lines to show.
    [int]$Tail = 50,

    # How long to wait for a component to become reachable.
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:DOTNET_NOLOGO = '1'

# --- Layout -----------------------------------------------------------------------------------

$RepoRoot = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $RepoRoot '.dev'
$LogDir = Join-Path $StateDir 'logs'

$ComposeFile = Join-Path $RepoRoot 'docker-compose.yml'
$EnvFile = Join-Path $RepoRoot '.env'
$EnvTemplate = Join-Path $RepoRoot '.env.example'
$SolutionFile = Join-Path $RepoRoot 'TokenMiner.sln'

$ApiProjectDir = Join-Path $RepoRoot 'backend/src/TokenMiner.Api'
$ApiDll = Join-Path $ApiProjectDir 'bin/Debug/net10.0/TokenMiner.Api.dll'
$InfraProject = Join-Path $RepoRoot 'backend/src/TokenMiner.Infrastructure'

$ProxyDir = Join-Path $RepoRoot 'stratum-proxy'
$ProxyExe = Join-Path $ProxyDir 'target/debug/sp-proxy.exe'

$MockPoolProject = Join-Path $RepoRoot 'mock-pool/src/MockPool/MockPool.csproj'
$MockPoolDll = Join-Path $RepoRoot 'mock-pool/src/MockPool/bin/Debug/net10.0/MockPool.dll'

$PortalDir = Join-Path $RepoRoot 'management-portal'

$ApiPort = 5210
$ProxyPort = 3333
$MockPoolPearlPort = 3335
$MockPoolQuantusPort = 3336
$PortalPort = 5273

# --- Output helpers ---------------------------------------------------------------------------

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Detail {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    [ok] $Message" -ForegroundColor Green
}

function Write-Skip {
    param([string]$Message)
    Write-Host "    [--] $Message" -ForegroundColor DarkYellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "    [!!] $Message" -ForegroundColor Red
}

# --- .env -------------------------------------------------------------------------------------

<#
    Parses .env into a hashtable, resolving ${VAR} references between keys. Compose does this
    substitution itself, but the natively-run API and proxy do not, so the values would otherwise
    drift apart when someone changes the port.
#>
function Read-DotEnv {
    param([string]$Path)

    $values = [ordered]@{}

    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()

        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }

        $separator = $trimmed.IndexOf('=')
        if ($separator -lt 1) { continue }

        $key = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        $values[$key] = $value
    }

    # Resolve ${OTHER_KEY} references. A few passes covers chained references.
    for ($pass = 0; $pass -lt 3; $pass++) {
        $changed = $false

        foreach ($key in @($values.Keys)) {
            $value = [string]$values[$key]

            foreach ($reference in @($values.Keys)) {
                $token = '${' + $reference + '}'

                if ($value.Contains($token)) {
                    $value = $value.Replace($token, [string]$values[$reference])
                }
            }

            if ($value -ne $values[$key]) {
                $values[$key] = $value
                $changed = $true
            }
        }

        if (-not $changed) { break }
    }

    return $values
}

function Initialize-Environment {
    if (-not (Test-Path $EnvFile)) {
        Copy-Item -Path $EnvTemplate -Destination $EnvFile
        Write-Ok "created .env from .env.example"
    }

    $settings = Read-DotEnv -Path $EnvFile

    $postgresPort = if ($settings.Contains('POSTGRES_PORT')) { $settings['POSTGRES_PORT'] } else { '5433' }
    $postgresUser = if ($settings.Contains('POSTGRES_USER')) { $settings['POSTGRES_USER'] } else { 'tokenminer' }
    $postgresPassword = if ($settings.Contains('POSTGRES_PASSWORD')) { $settings['POSTGRES_PASSWORD'] } else { 'tokenminer' }
    $postgresDb = if ($settings.Contains('POSTGRES_DB')) { $settings['POSTGRES_DB'] } else { 'tokenminer' }

    # The API is configured from the environment so .env stays the single source of truth, and so
    # the shared secret can never disagree with what the proxy is given.
    $env:ASPNETCORE_ENVIRONMENT = 'Development'
    $env:ConnectionStrings__Default = "Host=localhost;Port=$postgresPort;Database=$postgresDb;Username=$postgresUser;Password=$postgresPassword"

    if ($settings.Contains('Auth__Jwt__SigningKey') -and $settings['Auth__Jwt__SigningKey'] -notlike 'replace-*') {
        $env:Auth__Jwt__SigningKey = $settings['Auth__Jwt__SigningKey']
    }

    if ($settings.Contains('Mining__ServiceSharedSecret')) {
        $env:Mining__ServiceSharedSecret = $settings['Mining__ServiceSharedSecret']
    }

    # The proxy's secret: an explicit TOKENMINER_SERVICE_SECRET wins, otherwise reuse the API's.
    $serviceSecret = $null
    if ($settings.Contains('TOKENMINER_SERVICE_SECRET') -and $settings['TOKENMINER_SERVICE_SECRET'] -ne '') {
        $serviceSecret = $settings['TOKENMINER_SERVICE_SECRET']
    }
    elseif ($settings.Contains('MINING_SERVICE_SHARED_SECRET') -and $settings['MINING_SERVICE_SHARED_SECRET'] -ne '') {
        $serviceSecret = $settings['MINING_SERVICE_SHARED_SECRET']
    }

    if ($null -eq $serviceSecret) {
        # Fall back to the development value the API ships with, so a bare checkout still runs.
        $serviceSecret = 'dev-only-service-shared-secret-change-me'
        Write-Fail "MINING_SERVICE_SHARED_SECRET is not set in .env; using the development default."
    }

    $env:TOKENMINER_SERVICE_SECRET = $serviceSecret
    $env:Mining__ServiceSharedSecret = $serviceSecret

    return [pscustomobject]@{
        PostgresPort = $postgresPort
        ServiceSecret = $serviceSecret
    }
}

# --- Native mode ------------------------------------------------------------------------------

function Start-NativeProxy {
    $env:TOKENMINER_API_BASE_URL = "http://localhost:$ApiPort"
    $env:TOKENMINER_SERVICE_ID = 'stratum-proxy'
    $env:STRATUM_LISTEN_ADDR = "0.0.0.0:$ProxyPort"
    # Debug is where the verbatim relay is logged, both directions.
    $env:RUST_LOG = if ($Debug) { 'debug' } else { 'info' }

    Write-Step "Building the stratum proxy"

    $exitCode = Invoke-Native -FilePath 'cargo' -Arguments @('build', '--locked', '-p', 'sp-proxy') -WorkingDirectory $ProxyDir

    if ($exitCode -ne 0) { throw "cargo build failed with exit code $exitCode." }

    if (-not (Test-Path $ProxyExe)) {
        throw "Proxy binary not found at $ProxyExe"
    }

    Write-Ok "built"
}

# --- Process tracking -------------------------------------------------------------------------

function Get-TrackedPid {
    param([string]$Name)

    $pidFile = Join-Path $StateDir "$Name.pid"

    if (-not (Test-Path $pidFile)) { return $null }

    $text = (Get-Content -Path $pidFile -Raw).Trim()

    if ($text -notmatch '^\d+$') { return $null }

    return [int]$text
}

function Test-ComponentRunning {
    param([string]$Name)

    $processId = Get-TrackedPid -Name $Name

    if ($null -eq $processId) { return $false }

    return $null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue)
}

function Start-TrackedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory = $RepoRoot
    )

    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

    $stdout = Join-Path $LogDir "$Name.log"
    $stderr = Join-Path $LogDir "$Name.err.log"

    # An empty file stands in for stdin. Redirecting stdout/stderr forces Start-Process to create the
    # child without a shell, so it inherits this terminal's console — including its input handle.
    # A service that reads the console (Vite's keyboard shortcuts, for one) then grabs the keyboard
    # and the terminal stops responding. Pointing stdin at a file keeps that from happening.
    $stdin = Join-Path $StateDir "$Name.stdin"
    if (-not (Test-Path $stdin)) { New-Item -ItemType File -Path $stdin -Force | Out-Null }

    $startArguments = @{
        FilePath               = $FilePath
        WorkingDirectory       = $WorkingDirectory
        RedirectStandardInput  = $stdin
        RedirectStandardOutput = $stdout
        RedirectStandardError  = $stderr
        WindowStyle            = 'Hidden'
        PassThru               = $true
    }

    # Start-Process rejects an empty -ArgumentList, and joins the array with spaces without
    # quoting, so anything containing a space has to be quoted here.
    if ($null -ne $ArgumentList -and $ArgumentList.Count -gt 0) {
        $startArguments['ArgumentList'] = $ArgumentList | ForEach-Object {
            if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
        }
    }

    $process = Start-Process @startArguments

    Set-Content -Path (Join-Path $StateDir "$Name.pid") -Value $process.Id

    return $process
}

function Stop-TrackedProcess {
    param([string]$Name)

    $processId = Get-TrackedPid -Name $Name

    if ($null -eq $processId) { return $false }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue

    if ($null -ne $process) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Write-Ok "stopped $Name (pid $processId)"
    }

    Remove-Item -Path (Join-Path $StateDir "$Name.pid") -Force -ErrorAction SilentlyContinue

    return $true
}

<#
    Safety net for the case where a process outlived its pid file — the API started via
    `dotnet watch`, for instance, or an earlier run that crashed. Whatever holds the port is the
    thing we want gone.
#>
function Clear-Port {
    param(
        [int]$Port,
        [string]$Label
    )

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue

    foreach ($listener in $listeners) {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Ok "freed $Label (port $Port, pid $($listener.OwningProcess))"
    }
}

# --- Waiting ----------------------------------------------------------------------------------

function Wait-ForTcpPort {
    param(
        [int]$Port,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        $client = New-Object System.Net.Sockets.TcpClient

        try {
            $client.Connect('127.0.0.1', $Port)
            return $true
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
        finally {
            $client.Dispose()
        }
    }

    return $false
}

function Wait-ForHttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3

            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    return $false
}

# --- Docker -----------------------------------------------------------------------------------

<#
    Runs a native command with its output captured and the error preference relaxed.

    Necessary because docker writes progress and "no such object" notices to stderr, which the
    script-wide $ErrorActionPreference = 'Stop' would otherwise turn into a terminating error.
#>
function Invoke-NativeQuiet {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    try {
        $output = & $FilePath @Arguments 2>&1

        return [pscustomobject]@{
            ExitCode = $LASTEXITCODE
            Output = @($output)
        }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

<#
    Runs a native command whose output the operator should see, merging stderr into stdout.

    Both cargo and docker compose report normal progress on stderr; without this, PowerShell
    treats it as an error and aborts the run.
#>
function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $previousLocation = Get-Location

    if ($WorkingDirectory) {
        Set-Location -Path $WorkingDirectory
    }

    try {
        & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host "    $_" }

        return $LASTEXITCODE
    }
    finally {
        Set-Location -Path $previousLocation
        $ErrorActionPreference = $previousPreference
    }
}

function Assert-DockerRunning {
    $result = Invoke-NativeQuiet -FilePath 'docker' -Arguments @('info')

    if ($result.ExitCode -ne 0) {
        throw "Docker is not running. Start Docker Desktop and try again."
    }
}

function Invoke-Compose {
    param([string[]]$Arguments)

    $exitCode = Invoke-Native `
        -FilePath 'docker' `
        -Arguments (@('compose', '--file', $ComposeFile) + $Arguments) `
        -WorkingDirectory $RepoRoot

    if ($exitCode -ne 0) {
        throw "docker compose $($Arguments -join ' ') failed with exit code $exitCode."
    }
}

<#
    Uses `docker ps` rather than `docker inspect`: inspecting a container that does not exist
    exits non-zero and writes to stderr, whereas listing simply returns nothing.
#>
function Test-ContainerRunning {
    param([string]$Name)

    $result = Invoke-NativeQuiet -FilePath 'docker' -Arguments @(
        'ps', '--filter', "name=^/$Name$", '--filter', 'status=running', '--format', '{{.Names}}'
    )

    if ($result.ExitCode -ne 0) { return $false }

    return $result.Output -contains $Name
}

function Get-ContainerHealth {
    param([string]$Name)

    if (-not (Test-ContainerRunning -Name $Name)) { return $null }

    $result = Invoke-NativeQuiet -FilePath 'docker' -Arguments @(
        'inspect', '--format', '{{.State.Health.Status}}', $Name
    )

    if ($result.ExitCode -ne 0 -or $result.Output.Count -eq 0) { return $null }

    return $result.Output[0]
}

# --- Commands ---------------------------------------------------------------------------------

function Invoke-Migrate {
    Write-Step "Applying database migrations"

    Invoke-Native -FilePath 'dotnet' -Arguments @('tool', 'restore', '--verbosity', 'quiet') | Out-Null

    $exitCode = Invoke-Native -FilePath 'dotnet' -Arguments @(
        'ef', 'database', 'update',
        '--project', $InfraProject,
        '--startup-project', $ApiProjectDir
    )

    if ($exitCode -ne 0) { throw "dotnet ef database update failed with exit code $exitCode." }

    Write-Ok "schema up to date"
}

function Start-Postgres {
    Write-Step "Starting PostgreSQL"

    Assert-DockerRunning
    Invoke-Compose -Arguments @('up', '-d', 'postgres')

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if ((Get-ContainerHealth -Name 'tokenminer-postgres') -eq 'healthy') {
            Write-Ok "postgres ready on port $($script:Settings.PostgresPort)"
            return
        }

        Start-Sleep -Seconds 1
    }

    throw "PostgreSQL did not become healthy within $TimeoutSeconds seconds."
}

function Start-Api {
    if (Test-ComponentRunning -Name 'api') {
        Write-Skip "API already running (pid $(Get-TrackedPid -Name 'api'))"
        return
    }

    Write-Step "Building the solution"

    $exitCode = Invoke-Native -FilePath 'dotnet' -Arguments @('build', $SolutionFile, '--nologo', '--verbosity', 'quiet')

    if ($exitCode -ne 0) { throw "dotnet build failed with exit code $exitCode." }

    Write-Ok "build succeeded"

    if (-not (Test-Path $ApiDll)) {
        throw "API assembly not found at $ApiDll"
    }

    # The content root matters: run from the assembly's own directory and the app finds neither
    # appsettings.json nor appsettings.Development.json.
    $env:ASPNETCORE_CONTENTROOT = $ApiProjectDir
    $env:ASPNETCORE_URLS = if ($Containers) {
        # Bound beyond loopback so the containerised proxy can reach it.
        "http://0.0.0.0:$ApiPort"
    }
    else {
        "http://localhost:$ApiPort"
    }

    # Serilog reads its level from configuration, so env beats appsettings.json.
    if ($Debug) {
        $env:Serilog__MinimumLevel__Default = 'Debug'
    }
    else {
        Remove-Item Env:\Serilog__MinimumLevel__Default -ErrorAction SilentlyContinue
    }

    Write-Step "Starting the API"

    Start-TrackedProcess -Name 'api' -FilePath (Get-Command dotnet).Source -ArgumentList @($ApiDll) | Out-Null

    if (-not (Wait-ForHttpOk -Url "http://localhost:$ApiPort/health/ready" -TimeoutSeconds $TimeoutSeconds)) {
        throw "The API did not become ready. See $(Join-Path $LogDir 'api.err.log')."
    }

    Write-Ok "API listening on http://localhost:$ApiPort (pid $(Get-TrackedPid -Name 'api'))"
}

function Start-Proxy {
    if ($Containers) {
        Write-Step "Starting the stratum proxy and nginx (containers)"

        Invoke-Compose -Arguments @('--profile', 'full', 'up', '-d', 'stratum-proxy', 'nginx')

        if (-not (Wait-ForTcpPort -Port $ProxyPort -TimeoutSeconds $TimeoutSeconds)) {
            throw "nginx did not start listening on port $ProxyPort."
        }

        Write-Ok "stratum entry point on port $ProxyPort (nginx -> proxy)"
        return
    }

    if (Test-ComponentRunning -Name 'proxy') {
        Write-Skip "proxy already running (pid $(Get-TrackedPid -Name 'proxy'))"
        return
    }

    Start-NativeProxy

    Write-Step "Starting the stratum proxy"

    Start-TrackedProcess -Name 'proxy' -FilePath $ProxyExe -ArgumentList @() | Out-Null

    if (-not (Wait-ForTcpPort -Port $ProxyPort -TimeoutSeconds $TimeoutSeconds)) {
        throw "The proxy did not start listening on port $ProxyPort. See $(Join-Path $LogDir 'proxy.err.log')."
    }

    Write-Ok "stratum listening on port $ProxyPort (pid $(Get-TrackedPid -Name 'proxy'))"
}

function Start-MockPool {
    if (Test-ComponentRunning -Name 'mock-pool') {
        Write-Skip "mock pool already running (pid $(Get-TrackedPid -Name 'mock-pool'))"
        return
    }

    Write-Step "Building the mock pool"

    $exitCode = Invoke-Native -FilePath 'dotnet' -Arguments @(
        'build', $MockPoolProject, '--nologo', '--verbosity', 'quiet'
    )

    if ($exitCode -ne 0) { throw "dotnet build failed for the mock pool with exit code $exitCode." }

    if (-not (Test-Path $MockPoolDll)) {
        throw "Mock pool assembly not found at $MockPoolDll"
    }

    Write-Ok "built"

    # The mock pool listens on a port per coin; point a registered pool's stratumEndpoint here.
    $env:MOCK_POOL_PRL_ADDR = "0.0.0.0:$MockPoolPearlPort"
    $env:MOCK_POOL_QTC_ADDR = "0.0.0.0:$MockPoolQuantusPort"
    $env:MOCK_POOL_LOG_LEVEL = if ($Debug) { 'debug' } else { 'info' }

    Write-Step "Starting the mock pool"

    Start-TrackedProcess -Name 'mock-pool' -FilePath (Get-Command dotnet).Source -ArgumentList @($MockPoolDll) | Out-Null

    if (-not (Wait-ForTcpPort -Port $MockPoolPearlPort -TimeoutSeconds $TimeoutSeconds)) {
        throw "The mock pool did not start listening on port $MockPoolPearlPort. See $(Join-Path $LogDir 'mock-pool.err.log')."
    }

    Write-Ok "mock pool on ports $MockPoolPearlPort (PRL) and $MockPoolQuantusPort (QTC) (pid $(Get-TrackedPid -Name 'mock-pool'))"
}

function Get-NodePath {
    foreach ($candidate in @('node.exe', 'node')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue

        if ($null -ne $command) { return $command.Source }
    }

    throw "Node.js was not found on PATH. Install Node.js to run the management portal."
}

function Get-NpmPath {
    foreach ($candidate in @('npm.cmd', 'npm')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue

        if ($null -ne $command) { return $command.Source }
    }

    throw "npm was not found on PATH. Install Node.js to run the management portal."
}

function Start-Portal {
    if (Test-ComponentRunning -Name 'portal') {
        Write-Skip "management portal already running (pid $(Get-TrackedPid -Name 'portal'))"
        return
    }

    $node = Get-NodePath
    $vite = Join-Path $PortalDir 'node_modules/vite/bin/vite.js'

    if (-not (Test-Path $vite)) {
        Write-Step "Installing management portal dependencies"

        $exitCode = Invoke-Native -FilePath (Get-NpmPath) -Arguments @('install') -WorkingDirectory $PortalDir

        if ($exitCode -ne 0) { throw "npm install failed with exit code $exitCode." }

        if (-not (Test-Path $vite)) { throw "vite was not found at $vite after npm install." }

        Write-Ok "dependencies installed"
    }

    # vite.config.ts reads VITE_API_URL; the default already points at the API, this keeps it in
    # step if the API port changes.
    $env:VITE_API_URL = "http://localhost:$ApiPort"

    Write-Step "Starting the management portal"

    # Run vite through node directly so the tracked process id is the server itself and stopping it
    # needs no orphan cleanup.
    Start-TrackedProcess -Name 'portal' `
        -FilePath $node `
        -ArgumentList @($vite, '--port', "$PortalPort", '--strictPort') `
        -WorkingDirectory $PortalDir | Out-Null

    if (-not (Wait-ForHttpOk -Url "http://localhost:$PortalPort" -TimeoutSeconds $TimeoutSeconds)) {
        throw "The management portal did not become reachable on port $PortalPort. See $(Join-Path $LogDir 'portal.err.log')."
    }

    Write-Ok "management portal on http://localhost:$PortalPort (pid $(Get-TrackedPid -Name 'portal'))"
}

function Invoke-Up {
    $script:Settings = Initialize-Environment

    Write-Step "TokenMiner - local stack"

    Start-Postgres
    Invoke-Migrate
    Start-Api

    if ($MockPool) { Start-MockPool }

    Start-Proxy

    if ($Portal) { Start-Portal }

    Write-Host ''
    Write-Host '  Ready.' -ForegroundColor Green
    Write-Host ''
    Write-Host "    API       http://localhost:$ApiPort"
    Write-Host "    Swagger   http://localhost:$ApiPort/swagger"
    Write-Host "    Health    http://localhost:$ApiPort/health/ready"
    Write-Host "    Postgres  localhost:$($script:Settings.PostgresPort)"
    Write-Host "    Stratum   localhost:$ProxyPort   <- point your miner here"

    if ($MockPool) {
        Write-Host "    Mock PRL  localhost:$MockPoolPearlPort"
        Write-Host "    Mock QTC  localhost:$MockPoolQuantusPort"
    }

    if ($Portal) {
        Write-Host "    Portal    http://localhost:$PortalPort"
    }

    if (-not $Containers) {
        Write-Host ''
        Write-Detail "Start a session with POST /api/mining/start to get a miner command."
        Write-Detail "Set Mining:PublicStratumEndpoint to localhost:$ProxyPort so the command targets the proxy."
    }

    if ($MockPool) {
        Write-Detail "Register a pool whose stratumEndpoint is localhost:$MockPoolPearlPort (PRL) or localhost:$MockPoolQuantusPort (QTC) to route through the mock."
    }

    Write-Host ''
    Write-Detail "logs:  ./scripts/dev.ps1 logs -Component api -Follow"
    Write-Detail "stop:  ./scripts/dev.ps1 down"
    Write-Host ''
}

<#
    Stops a single component so 'restart -Service <name>' can bring just that piece back.
    Mirrors what Invoke-Down does, narrowed to one name.
#>
function Stop-Component {
    param([string]$Name)

    switch ($Name) {
        'postgres' {
            if (Test-ContainerRunning -Name 'tokenminer-postgres') {
                Invoke-Compose -Arguments @('stop', 'postgres')
                Write-Ok "stopped postgres"
            }
        }
        'api' {
            Stop-TrackedProcess -Name 'api' | Out-Null
            Clear-Port -Port $ApiPort -Label 'api'
        }
        'proxy' {
            if (Test-ContainerRunning -Name 'tokenminer-stratum-proxy') {
                Invoke-Compose -Arguments @('--profile', 'full', 'rm', '-s', '-f', 'stratum-proxy', 'nginx')
                Write-Ok "stopped stratum-proxy and nginx"
            }
            else {
                Stop-TrackedProcess -Name 'proxy' | Out-Null
                Clear-Port -Port $ProxyPort -Label 'stratum'
            }
        }
        'mock-pool' {
            Stop-TrackedProcess -Name 'mock-pool' | Out-Null
            Clear-Port -Port $MockPoolPearlPort -Label 'mock-pool (prl)'
            Clear-Port -Port $MockPoolQuantusPort -Label 'mock-pool (qtc)'
        }
        'portal' {
            Stop-TrackedProcess -Name 'portal' | Out-Null
            Clear-Port -Port $PortalPort -Label 'portal'
        }
    }
}

function Invoke-Restart {
    if ($Service -eq 'all') {
        Invoke-Down
        Invoke-Up
        return
    }

    $script:Settings = Initialize-Environment

    Write-Step "Restarting $Service"

    Stop-Component -Name $Service

    switch ($Service) {
        'postgres' { Start-Postgres }
        'api' { Start-Api }
        'proxy' { Start-Proxy }
        'mock-pool' { Start-MockPool }
        'portal' { Start-Portal }
    }

    Write-Host ''
    Write-Host "  Restarted $Service." -ForegroundColor Green
    Write-Host ''
}

function Invoke-Down {
    Write-Step "Stopping the stack"

    if (Test-ContainerRunning -Name 'tokenminer-stratum-proxy') {
        Invoke-Compose -Arguments @('--profile', 'full', 'rm', '-s', '-f', 'stratum-proxy', 'nginx')
        Write-Ok "stopped stratum-proxy and nginx"
    }

    Stop-TrackedProcess -Name 'portal' | Out-Null
    Stop-TrackedProcess -Name 'mock-pool' | Out-Null
    Stop-TrackedProcess -Name 'proxy' | Out-Null
    Stop-TrackedProcess -Name 'api' | Out-Null

    # Catch anything that outlived its pid file before declaring the ports free.
    Clear-Port -Port $PortalPort -Label 'portal'
    Clear-Port -Port $MockPoolPearlPort -Label 'mock-pool (prl)'
    Clear-Port -Port $MockPoolQuantusPort -Label 'mock-pool (qtc)'
    Clear-Port -Port $ProxyPort -Label 'stratum'
    Clear-Port -Port $ApiPort -Label 'api'

    if (Test-ContainerRunning -Name 'tokenminer-postgres') {
        $composeArguments = @('down')

        if ($Purge) {
            $composeArguments += '--volumes'
        }

        Invoke-Compose -Arguments $composeArguments

        if ($Purge) {
            Write-Ok "postgres stopped and its volume deleted"
        }
        else {
            Write-Ok "postgres stopped (volume kept; pass -Purge to delete it)"
        }
    }

    Write-Host ''
    Write-Host '  Stopped.' -ForegroundColor Green
    Write-Host ''
}

function Invoke-Status {
    Write-Step "TokenMiner - local stack"

    if (Test-ContainerRunning -Name 'tokenminer-postgres') {
        $health = Get-ContainerHealth -Name 'tokenminer-postgres'
        Write-Ok "postgres        running ($health)"
    }
    else {
        Write-Fail "postgres        not running"
    }

    if (Test-ComponentRunning -Name 'api') {
        $processId = Get-TrackedPid -Name 'api'
        $ready = Wait-ForHttpOk -Url "http://localhost:$ApiPort/health/ready" -TimeoutSeconds 3

        if ($ready) {
            Write-Ok "api             running (pid $processId, ready)"
        }
        else {
            Write-Fail "api             running (pid $processId) but not ready"
        }
    }
    elseif (Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue) {
        Write-Fail "api             port $ApiPort held by an untracked process"
    }
    else {
        Write-Fail "api             not running"
    }

    if ($Containers -or (Test-ContainerRunning -Name 'tokenminer-stratum-proxy')) {
        if (Test-ContainerRunning -Name 'tokenminer-stratum-proxy') {
            Write-Ok "stratum-proxy   running (container)"
        }
        else {
            Write-Fail "stratum-proxy   not running"
        }

        if (Test-ContainerRunning -Name 'tokenminer-nginx') {
            Write-Ok "nginx           running (container)"
        }
        else {
            Write-Fail "nginx           not running"
        }
    }
    elseif (Test-ComponentRunning -Name 'proxy') {
        Write-Ok "stratum-proxy   running (pid $(Get-TrackedPid -Name 'proxy'))"
    }
    elseif (Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue) {
        Write-Fail "stratum-proxy   port $ProxyPort held by an untracked process"
    }
    else {
        Write-Fail "stratum-proxy   not running"
    }

    if ($MockPool -or (Test-ComponentRunning -Name 'mock-pool')) {
        if (Test-ComponentRunning -Name 'mock-pool') {
            Write-Ok "mock-pool       running (pid $(Get-TrackedPid -Name 'mock-pool'))"
        }
        elseif (Get-NetTCPConnection -LocalPort $MockPoolPearlPort -State Listen -ErrorAction SilentlyContinue) {
            Write-Fail "mock-pool       port $MockPoolPearlPort held by an untracked process"
        }
        else {
            Write-Fail "mock-pool       not running"
        }
    }

    if ($Portal -or (Test-ComponentRunning -Name 'portal')) {
        if (Test-ComponentRunning -Name 'portal') {
            Write-Ok "portal          running (pid $(Get-TrackedPid -Name 'portal'))"
        }
        elseif (Get-NetTCPConnection -LocalPort $PortalPort -State Listen -ErrorAction SilentlyContinue) {
            Write-Fail "portal          port $PortalPort held by an untracked process"
        }
        else {
            Write-Fail "portal          not running"
        }
    }

    Write-Host ''
}

function Invoke-Logs {
    $names = switch ($Component) {
        'api' { @('api') }
        'proxy' { @('proxy') }
        'mock-pool' { @('mock-pool') }
        'portal' { @('portal') }
        default { @('api', 'proxy', 'mock-pool', 'portal') }
    }

    if ($Follow -and $names.Count -gt 1) {
        Write-Fail "-Follow needs a single -Component (api, proxy, mock-pool or portal); showing recent lines only."
        $Follow = $false
    }

    foreach ($name in $names) {
        $path = Join-Path $LogDir "$name.log"

        Write-Host ''
        Write-Host "--- $name ($path)" -ForegroundColor Cyan

        if (-not (Test-Path $path)) {
            Write-Skip "no log yet"
            continue
        }

        Get-Content -Path $path -Tail $Tail

        if ($Follow) {
            Get-Content -Path $path -Tail 0 -Wait
        }
    }
}

function Invoke-Help {
    Write-Host ''
    Write-Host '  TokenMiner dev stack' -ForegroundColor Green
    Write-Host ''
    Write-Host '  Usage:' -ForegroundColor Cyan
    Write-Detail './scripts/dev.ps1 [command] [options]'
    Write-Host ''
    Write-Host '  Commands:' -ForegroundColor Cyan
    Write-Detail 'up        Bring the stack up (default)'
    Write-Detail 'down      Stop the stack'
    Write-Detail 'restart   Stop, then bring the stack up again (or one service with -Service)'
    Write-Detail 'status    Show what is running'
    Write-Detail 'logs      Print (and optionally follow) component logs'
    Write-Detail 'migrate   Apply database migrations'
    Write-Detail 'help      Show this message'
    Write-Host ''
    Write-Host '  Options:' -ForegroundColor Cyan
    Write-Detail '-Containers          Run the proxy and nginx from docker compose'
    Write-Detail '-MockPool            Also start the mock Stratum pool (Pearl 3335, Quantus 3336)'
    Write-Detail '-Portal              Also start the management portal (port 5273)'
    Write-Detail '-Purge               With down: also delete the PostgreSQL volume'
    Write-Detail '-Service <name>      With restart: postgres, api, proxy, mock-pool or portal (default all)'
    Write-Detail '-Component <name>    With logs: api, proxy, mock-pool, portal or all'
    Write-Detail '-Follow              With logs: keep reading as new lines arrive (one component)'
    Write-Detail '-Debug               Raise every component log level to debug'
    Write-Detail '-Tail <n>            With logs: how many existing lines to show (default 50)'
    Write-Detail '-TimeoutSeconds <n>  How long to wait for a component to start (default 120)'
    Write-Host ''
    Write-Host '  Examples:' -ForegroundColor Cyan
    Write-Detail './scripts/dev.ps1 up -MockPool -Portal'
    Write-Detail './scripts/dev.ps1 up -Debug'
    Write-Detail './scripts/dev.ps1 restart -Service api'
    Write-Detail './scripts/dev.ps1 logs -Component api -Follow'
    Write-Detail './scripts/dev.ps1 down -Purge'
    Write-Host ''
}

# --- Entry point ------------------------------------------------------------------------------

# The script is deliberately not an advanced function (see -Debug), so PowerShell would otherwise
# treat an unknown argument as positional and silently ignore it.
if ($args.Count -gt 0) {
    Write-Host ''
    Write-Fail "Unknown argument(s): $($args -join ' '). Run './scripts/dev.ps1 help' for usage."
    Write-Host ''
    exit 1
}

Push-Location $RepoRoot

try {
    switch ($Command) {
        'up' { Invoke-Up }
        'down' { Invoke-Down }
        'restart' { Invoke-Restart }
        'status' { Invoke-Status }
        'logs' { Invoke-Logs }
        'migrate' { $script:Settings = Initialize-Environment; Start-Postgres; Invoke-Migrate }
        'help' { Invoke-Help }
    }
}
catch {
    Write-Host ''
    Write-Fail $_.Exception.Message
    Write-Host ''
    exit 1
}
finally {
    Pop-Location
}
