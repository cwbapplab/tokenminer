namespace TokenMiner.Application.Common.Exceptions;

/// <summary>
/// Base type for expected, client-facing failures. The API maps <see cref="StatusCode"/>
/// onto the HTTP response, so handlers never reference HTTP concerns directly.
/// </summary>
public abstract class AppException(string message) : Exception(message)
{
    public abstract int StatusCode { get; }
}

public sealed class BadRequestException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status400BadRequest;
}

public sealed class ConflictException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status409Conflict;
}

public sealed class UnauthorizedException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status401Unauthorized;
}

public sealed class ForbiddenException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status403Forbidden;
}

public sealed class NotFoundException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status404NotFound;
}

public sealed class TooManyRequestsException(string message) : AppException(message)
{
    public override int StatusCode => StatusCodes.Status429TooManyRequests;
}

/// <summary>Minimal HTTP status constants so the Application layer needs no ASP.NET dependency.</summary>
file static class StatusCodes
{
    public const int Status400BadRequest = 400;
    public const int Status401Unauthorized = 401;
    public const int Status403Forbidden = 403;
    public const int Status404NotFound = 404;
    public const int Status409Conflict = 409;
    public const int Status429TooManyRequests = 429;
}
