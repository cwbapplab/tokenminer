import { useEffect, useMemo, useState } from 'react';
import Chart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';
import { Icon } from '../components/icons';
import { DataTable, ErrorState, Loading, StatCell, useAsyncList } from '../components/ui';
import { api } from '../lib/api';
import { useLookups } from '../lib/lookups';
import { coinCode } from '../lib/lookups';
import { formatDateTime, formatInt, formatRelative, formatUsd, humanize } from '../lib/format';
import type { Column, Row } from '../lib/resources';
import type { ConversionTransaction, PoolPayout, SystemStatus } from '../lib/types';

const CHART_COLORS = ['#f73a0b', '#2769ee', '#22bc32', '#eeac27', '#f34040', '#8b8b8b'];

const baseChartOptions: ApexOptions = {
  chart: {
    toolbar: { show: false },
    fontFamily: 'Poppins, sans-serif',
    foreColor: '#8b8b8b',
  },
  dataLabels: { enabled: false },
  grid: { borderColor: '#ececec', strokeDashArray: 4 },
  tooltip: { theme: 'light' },
};

function dayKey(value: string): string {
  const date = new Date(value);

  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function shortDay(key: string): string {
  const [, month, day] = key.split('-');

  return `${day}/${month}`;
}

function useStatus(): { status: SystemStatus | null; loading: boolean; error: string | null; reload: () => void } {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .get<SystemStatus>('/api/admin/system/status')
      .then((result) => {
        if (!cancelled) {
          setStatus(result);
          setError(null);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : 'Could not load the system status.');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [nonce]);

  return { status, loading, error, reload: () => setNonce((value) => value + 1) };
}

export function DashboardPage() {
  const systemStatus = useStatus();
  const payouts = useAsyncList<PoolPayout>('/api/admin/pool-payouts?limit=200');
  const conversions = useAsyncList<ConversionTransaction>('/api/admin/conversions?limit=200');
  const lookups = useLookups();

  const payoutChart = useMemo(() => {
    // Fourteen days is enough to see a trend without the axis turning to noise.
    const days: string[] = [];
    const today = new Date();

    for (let offset = 13; offset >= 0; offset -= 1) {
      const date = new Date(today);
      date.setDate(today.getDate() - offset);
      days.push(dayKey(date.toISOString()));
    }

    const totals = new Map<string, number>();
    let redeemed = 0;
    let outstanding = 0;

    for (const payout of payouts.data) {
      const key = dayKey(payout.receivedAt ?? payout.createdAt);
      totals.set(key, (totals.get(key) ?? 0) + payout.amount);

      redeemed += payout.redeemedAmount;
      outstanding += payout.amount - payout.redeemedAmount;
    }

    return {
      categories: days.map(shortDay),
      data: days.map((day) => Number((totals.get(day) ?? 0).toFixed(4))),
      redeemed,
      outstanding,
    };
  }, [payouts.data]);

  const conversionChart = useMemo(() => {
    const counts = new Map<string, number>();

    for (const conversion of conversions.data) {
      counts.set(conversion.status, (counts.get(conversion.status) ?? 0) + 1);
    }

    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);

    return {
      labels: entries.map(([status]) => humanize(status)),
      series: entries.map(([, count]) => count),
    };
  }, [conversions.data]);

  const activityColumns: Column[] = [
    { key: 'kind', label: 'Event', format: 'strong' },
    { key: 'detail', label: 'Detail' },
    { key: 'status', label: 'Status', format: 'badge' },
    { key: 'at', label: 'When' },
  ];

  const activity: Row[] = useMemo(() => {
    const fromPayouts: Row[] = payouts.data.map((payout) => ({
      id: `payout-${payout.id}`,
      kind: 'Pool payout',
      detail: `${payout.amount} ${coinCode(lookups.coins, payout.coinId)} · ${payout.transactionHash ?? 'awaiting hash'}`,
      status: payout.status,
      at: payout.receivedAt ?? payout.createdAt,
    }));

    const fromConversions: Row[] = conversions.data.map((conversion) => ({
      id: `conversion-${conversion.id}`,
      kind: 'Conversion',
      detail: `${conversion.sourceAmount} ${conversion.sourceCoinCode} → ${conversion.destinationCoinCode}`,
      status: conversion.status,
      at: conversion.createdAt,
    }));

    return [...fromPayouts, ...fromConversions]
      .sort((a, b) => new Date(String(b.at)).getTime() - new Date(String(a.at)).getTime())
      .slice(0, 8)
      .map((row) => ({ ...row, at: formatDateTime(String(row.at)) }));
  }, [payouts.data, conversions.data, lookups.coins]);

  if (systemStatus.loading) {
    return <Loading label="Loading the dashboard" />;
  }

  if (systemStatus.error) {
    return <ErrorState message={systemStatus.error} onRetry={systemStatus.reload} />;
  }

  const status = systemStatus.status;

  const areaOptions: ApexOptions = {
    ...baseChartOptions,
    chart: { ...baseChartOptions.chart, type: 'area', height: 300 },
    colors: [CHART_COLORS[0]],
    stroke: { curve: 'smooth', width: 2 },
    fill: { type: 'gradient', gradient: { opacityFrom: 0.35, opacityTo: 0.05 } },
    xaxis: { categories: payoutChart.categories, labels: { style: { fontSize: '11px' } } },
    yaxis: { labels: { formatter: (value: number) => value.toFixed(2) } },
  };

  const donutOptions: ApexOptions = {
    ...baseChartOptions,
    labels: conversionChart.labels,
    colors: CHART_COLORS,
    legend: { position: 'bottom', fontSize: '12px' },
    stroke: { width: 0 },
  };

  const hasConversions = conversionChart.series.length > 0;

  return (
    <>
      <div className="row tm-card-row">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <div className="stat-strip">
                <StatCell
                  label="Running sessions"
                  value={formatInt(status?.runningSessions)}
                  icon="activity"
                  tone="success"
                  hint={status?.pausedSessions ? `${status.pausedSessions} paused` : 'No paused sessions'}
                />
                <StatCell
                  label="Provider balance"
                  value={formatUsd(status?.providerBalance ?? null)}
                  icon="wallet"
                  hint={
                    status?.providerBalanceCheckedAt
                      ? `checked ${formatRelative(status.providerBalanceCheckedAt)}`
                      : 'never checked'
                  }
                />
                <StatCell
                  label="Pending shares"
                  value={formatInt(status?.pendingShares)}
                  icon="cpu"
                  tone={status && status.pendingShares > 0 ? 'warning' : 'success'}
                  hint="waiting on the reward processor"
                />
                <StatCell
                  label="In flight"
                  value={formatInt((status?.conversionsInFlight ?? 0) + (status?.providerDepositsInFlight ?? 0))}
                  icon="layers"
                  tone="info"
                  hint={`${status?.conversionsInFlight ?? 0} conversions · ${status?.providerDepositsInFlight ?? 0} deposits`}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row tm-card-row">
        <div className="col-xl-8">
          <div className="card">
            <div className="card-header">
              <div>
                <h4 className="card-title">Pool payouts</h4>
                <div className="field-hint">Mined value received per day, across every pool.</div>
              </div>
              <div className="text-end">
                <div className="cell-strong">{formatUsd(payoutChart.redeemed, 4)}</div>
                <div className="field-hint mb-0">attributed to shares</div>
              </div>
            </div>
            <div className="card-body pt-0">
              {payouts.loading && <div className="spinner lg mx-auto" />}
              {payouts.error && <div className="alert alert-danger">{payouts.error}</div>}
              {!payouts.loading && !payouts.error && (
                <div className="chart-host">
                  <Chart options={areaOptions} series={[{ name: 'Payout', data: payoutChart.data }]} type="area" height={300} />
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="col-xl-4">
          <div className="card">
            <div className="card-header">
              <div>
                <h4 className="card-title">Conversions</h4>
                <div className="field-hint">By state, over the last 200 transactions.</div>
              </div>
            </div>
            <div className="card-body pt-0">
              {conversions.loading && <div className="spinner lg mx-auto" />}
              {conversions.error && <div className="alert alert-danger">{conversions.error}</div>}
              {!conversions.loading && !conversions.error && hasConversions && (
                <div className="chart-host">
                  <Chart options={donutOptions} series={conversionChart.series} type="donut" height={300} />
                </div>
              )}
              {!conversions.loading && !conversions.error && !hasConversions && (
                <div className="tm-placeholder">
                  <div className="placeholder-icon">
                    <Icon name="swap" />
                  </div>
                  <h5>No conversions yet</h5>
                  <p>Once a pool payout settles, the conversion pipeline picks it up here.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="row tm-card-row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <div>
                <h4 className="card-title">Recent activity</h4>
                <div className="field-hint">The newest payouts and conversions, merged by time.</div>
              </div>
            </div>
            <div className="card-body pt-0">
              <DataTable
                columns={activityColumns}
                rows={activity}
                lookups={lookups}
                loading={payouts.loading || conversions.loading}
                error={payouts.error ?? conversions.error}
                onRetry={() => {
                  payouts.reload();
                  conversions.reload();
                }}
                emptyTitle="Nothing has happened yet"
                emptyMessage="Payouts and conversions appear here as the treasury pipeline runs."
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
