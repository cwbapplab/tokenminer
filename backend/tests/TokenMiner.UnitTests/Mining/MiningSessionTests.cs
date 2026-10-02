using FluentAssertions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Mining;

public sealed class UserHardwareMinerTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static UserHardwareMiner CreateSession() => new(
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        Guid.NewGuid(),
        "user-hardware",
        Now);

    [Fact]
    public void NewSession_IsRunning()
    {
        var session = CreateSession();

        session.Status.Should().Be(MiningSessionStatus.Running);
        session.IsActive.Should().BeTrue();
        session.StartedAt.Should().Be(Now);
        session.LastActivityAt.Should().Be(Now);
        session.PausedAt.Should().BeNull();
        session.StoppedAt.Should().BeNull();
    }

    [Fact]
    public void Pause_TransitionsOnlyOnce()
    {
        var session = CreateSession();
        var pausedAt = Now.AddMinutes(1);

        session.Pause(MiningStopReasonDbValues.ConnectionFailure, pausedAt).Should().BeTrue();
        session.Pause(MiningStopReasonDbValues.ConnectionFailure, pausedAt.AddMinutes(1)).Should().BeFalse();

        session.Status.Should().Be(MiningSessionStatus.Paused);
        session.PauseReason.Should().Be("connection-failure");
        session.PausedAt.Should().Be(pausedAt);
        session.IsActive.Should().BeTrue();
    }

    [Fact]
    public void Resume_OnlyAppliesWhenPausedAndClearsTheReason()
    {
        var session = CreateSession();

        // Not paused yet.
        session.Resume(Now.AddMinutes(1)).Should().BeFalse();

        session.Pause(MiningStopReasonDbValues.ConnectionFailure, Now.AddMinutes(1));
        var resumedAt = Now.AddMinutes(2);

        session.Resume(resumedAt).Should().BeTrue();

        session.Status.Should().Be(MiningSessionStatus.Running);
        session.PauseReason.Should().BeNull();
        session.PausedAt.Should().BeNull();
        session.LastActivityAt.Should().Be(resumedAt);
    }

    [Fact]
    public void Stop_TransitionsFromRunningOrPausedExactlyOnce()
    {
        var session = CreateSession();
        var stoppedAt = Now.AddMinutes(5);

        session.Stop(MiningStopReasonDbValues.UserTriggered, stoppedAt).Should().BeTrue();
        session.Stop(MiningStopReasonDbValues.UserTriggered, stoppedAt.AddMinutes(1)).Should().BeFalse();

        session.Status.Should().Be(MiningSessionStatus.Stopped);
        session.StopReason.Should().Be("user-triggered");
        session.StoppedAt.Should().Be(stoppedAt);
        session.IsActive.Should().BeFalse();
    }

    [Fact]
    public void Touch_OnlyMovesLastActivity()
    {
        var session = CreateSession();
        var later = Now.AddSeconds(30);

        session.Touch(later);

        session.LastActivityAt.Should().Be(later);
        session.StartedAt.Should().Be(Now);
        session.Status.Should().Be(MiningSessionStatus.Running);
    }
}

public sealed class MiningSessionEnumTests
{
    [Theory]
    [InlineData(MiningSessionStatus.Running, "running")]
    [InlineData(MiningSessionStatus.Paused, "paused")]
    [InlineData(MiningSessionStatus.Stopped, "stopped")]
    public void SessionStatus_RoundTrips(MiningSessionStatus status, string expected)
    {
        status.ToDbValue().Should().Be(expected);
        MiningSessionStatusDbValues.FromDbValue(expected).Should().Be(status);
    }

    [Fact]
    public void ActiveStatusesAreRunningAndPaused()
    {
        MiningSessionStatusDbValues.Active.Should().BeEquivalentTo(["running", "paused"]);
    }

    [Theory]
    [InlineData(MiningStopReason.ConnectionFailure, "connection-failure")]
    [InlineData(MiningStopReason.DriverCrash, "driver-crash")]
    [InlineData(MiningStopReason.MinerClosed, "miner-closed")]
    [InlineData(MiningStopReason.UserTriggered, "user-triggered")]
    [InlineData(MiningStopReason.Unknown, "unknown")]
    public void StopReason_RoundTrips(MiningStopReason reason, string expected)
    {
        reason.ToDbValue().Should().Be(expected);
        MiningStopReasonDbValues.FromDbValue(expected).Should().Be(reason);
    }

    [Theory]
    [InlineData("user-triggered", true)]
    [InlineData("connection-failure", true)]
    [InlineData("user_triggered", false)]
    [InlineData("nope", false)]
    [InlineData(null, false)]
    public void StopReason_TryParseOnlyAcceptsDocumentedValues(string? value, bool expected)
    {
        MiningStopReasonDbValues.TryParseDbValue(value, out _).Should().Be(expected);
    }

    [Theory]
    [InlineData(MiningEventType.MiningStarted, "mining_started")]
    [InlineData(MiningEventType.MiningPaused, "mining_paused")]
    [InlineData(MiningEventType.MiningResumed, "mining_resumed")]
    [InlineData(MiningEventType.MiningStopped, "mining_stopped")]
    [InlineData(MiningEventType.ConnectionFailure, "connection_failure")]
    public void EventType_RoundTrips(MiningEventType eventType, string expected)
    {
        eventType.ToDbValue().Should().Be(expected);
        MiningEventTypeDbValues.FromDbValue(expected).Should().Be(eventType);
    }
}
