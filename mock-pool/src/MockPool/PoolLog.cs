using System.Globalization;

namespace MockPool;

public enum LogVerbosity
{
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
}

/// <summary>Minimal console logger. Keeps the mock pool free of extra dependencies.</summary>
public static class PoolLog
{
    private static LogVerbosity _level = LogVerbosity.Info;

    public static void SetLevel(LogVerbosity level) => _level = level;

    public static void Debug(string message) => Write(LogVerbosity.Debug, "DEBUG", message);

    public static void Info(string message) => Write(LogVerbosity.Info, "INFO", message);

    public static void Warn(string message) => Write(LogVerbosity.Warning, "WARN", message);

    public static void Error(string message) => Write(LogVerbosity.Error, "ERROR", message);

    private static void Write(LogVerbosity level, string label, string message)
    {
        if (level < _level)
        {
            return;
        }

        var stamp = DateTimeOffset.UtcNow.ToString("HH:mm:ss.fff", CultureInfo.InvariantCulture);
        Console.Out.WriteLine($"[{stamp}] {label,-5} {message}");
    }
}
