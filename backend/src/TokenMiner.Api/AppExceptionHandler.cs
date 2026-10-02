using FluentValidation;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using TokenMiner.Application.Common.Exceptions;

namespace TokenMiner.Api;

/// <summary>
/// Translates expected application failures and validation errors into problem responses,
/// so handlers and endpoints never build error payloads themselves.
/// </summary>
internal sealed class AppExceptionHandler(ILogger<AppExceptionHandler> logger) : IExceptionHandler
{
    public async ValueTask<bool> TryHandleAsync(
        HttpContext httpContext,
        Exception exception,
        CancellationToken cancellationToken)
    {
        var (statusCode, title, errors) = Map(exception);

        if (statusCode is null)
        {
            return false;
        }

        if (statusCode >= StatusCodes.Status500InternalServerError)
        {
            logger.LogError(exception, "Unhandled application error.");
        }
        else
        {
            logger.LogInformation("Request failed with {StatusCode}: {Message}", statusCode, title);
        }

        var problem = new ProblemDetails
        {
            Status = statusCode,
            Title = title,
            Type = $"https://httpstatuses.io/{statusCode}",
            Instance = httpContext.Request.Path,
        };

        if (errors is not null)
        {
            problem.Extensions["errors"] = errors;
        }

        httpContext.Response.StatusCode = statusCode.Value;
        await httpContext.Response.WriteAsJsonAsync(problem, cancellationToken);

        return true;
    }

    private static (int? StatusCode, string Title, IDictionary<string, string[]>? Errors) Map(Exception exception) =>
        exception switch
        {
            ValidationException validation => (
                StatusCodes.Status400BadRequest,
                "Validation failed.",
                validation.Errors
                    .GroupBy(failure => failure.PropertyName)
                    .ToDictionary(
                        group => group.Key,
                        group => group.Select(failure => failure.ErrorMessage).ToArray())),

            // Thrown by minimal APIs for an unreadable/invalid body. Development turns this
            // into an exception rather than a 400, so map it explicitly.
            BadHttpRequestException badRequest => (badRequest.StatusCode, "Malformed request body.", null),

            AppException appException => (appException.StatusCode, appException.Message, null),

            _ => (null, string.Empty, null),
        };
}
