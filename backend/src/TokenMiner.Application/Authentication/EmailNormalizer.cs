namespace TokenMiner.Application.Authentication;

public static class EmailNormalizer
{
    /// <summary>
    /// Canonical form used for uniqueness and lookups. Email is case-insensitive in
    /// practice for the providers we support, so it is upper-cased for comparison.
    /// </summary>
    public static string Normalize(string email) => email.Trim().ToUpperInvariant();
}
