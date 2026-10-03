using MockPool.Stratum;
using Xunit;

namespace MockPool.Tests;

public sealed class ShareVerdictTests
{
    [Theory]
    [InlineData("00aabbcc", StratumVerdict.Accepted)]
    [InlineData("20aabbcc", StratumVerdict.Other)]
    [InlineData("21aabbcc", StratumVerdict.JobNotFound)]
    [InlineData("22aabbcc", StratumVerdict.Duplicate)]
    [InlineData("23aabbcc", StratumVerdict.LowDifficulty)]
    [InlineData("24aabbcc", StratumVerdict.Unauthorized)]
    [InlineData("25aabbcc", StratumVerdict.NotSubscribed)]
    [InlineData("26aabbcc", StratumVerdict.InvalidProof)]
    [InlineData("ffaabbcc", StratumVerdict.Rejected)]
    [InlineData("FFAABBCC", StratumVerdict.Rejected)]
    public void Nonce_prefix_selects_the_verdict(string nonce, StratumVerdict expected) =>
        Assert.Equal(expected, ShareVerdicts.Resolve("PRL-1", nonce));

    [Fact]
    public void Unknown_nonce_prefix_defaults_to_accepted() =>
        Assert.Equal(StratumVerdict.Accepted, ShareVerdicts.Resolve("PRL-1", "aabbccdd"));

    [Theory]
    [InlineData("stale-1")]
    [InlineData("stale")]
    [InlineData("Stale-Job")]
    [InlineData("invalid-job")]
    public void Stale_job_is_rejected_when_the_nonce_has_no_tag(string jobId) =>
        Assert.Equal(StratumVerdict.JobNotFound, ShareVerdicts.Resolve(jobId, "aabbccdd"));

    [Fact]
    public void Nonce_tag_takes_precedence_over_the_job_id() =>
        Assert.Equal(StratumVerdict.LowDifficulty, ShareVerdicts.Resolve("stale-1", "23000000"));

    [Fact]
    public void Explicit_accept_tag_overrides_a_stale_job() =>
        Assert.Equal(StratumVerdict.Accepted, ShareVerdicts.Resolve("stale-1", "00000000"));

    [Fact]
    public void Superseded_job_is_stale_when_the_nonce_has_no_tag() =>
        Assert.Equal(StratumVerdict.JobNotFound, ShareVerdicts.Resolve("PRL-1", "aabbccdd", jobIsSuperseded: true));

    [Fact]
    public void Nonce_tag_still_wins_over_a_superseded_job() =>
        Assert.Equal(StratumVerdict.LowDifficulty, ShareVerdicts.Resolve("PRL-1", "23000000", jobIsSuperseded: true));

    [Fact]
    public void Explicit_accept_tag_overrides_a_superseded_job() =>
        Assert.Equal(StratumVerdict.Accepted, ShareVerdicts.Resolve("PRL-1", "00000000", jobIsSuperseded: true));

    [Theory]
    [InlineData(StratumVerdict.LowDifficulty, 23)]
    [InlineData(StratumVerdict.Duplicate, 22)]
    [InlineData(StratumVerdict.Unauthorized, 24)]
    [InlineData(StratumVerdict.NotSubscribed, 25)]
    [InlineData(StratumVerdict.JobNotFound, 21)]
    [InlineData(StratumVerdict.Other, 20)]
    public void Verdicts_map_to_the_standard_error_codes(StratumVerdict verdict, int code)
    {
        Assert.True(StratumErrors.TryFor(verdict, out var error));
        Assert.Equal(code, error.Code);
    }

    [Theory]
    [InlineData(StratumVerdict.Accepted)]
    [InlineData(StratumVerdict.Rejected)]
    public void Accepted_and_bare_rejection_have_no_error(StratumVerdict verdict) =>
        Assert.False(StratumErrors.TryFor(verdict, out _));

    [Theory]
    [InlineData(StratumVerdict.JobNotFound, 21, "stale job")]
    [InlineData(StratumVerdict.Duplicate, 22, "duplicate share")]
    [InlineData(StratumVerdict.LowDifficulty, 23, "low difficulty share")]
    [InlineData(StratumVerdict.InvalidProof, 26, "invalid proof")]
    [InlineData(StratumVerdict.Unauthorized, 27, "unauthorized")]
    public void Pearl_verdicts_map_to_the_pearl_error_codes(StratumVerdict verdict, int code, string message)
    {
        Assert.True(PearlErrors.TryFor(verdict, out var error));
        Assert.Equal(code, error.Code);
        Assert.Equal(message, error.Message);
    }
}
