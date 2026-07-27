# Phase 10 Monitoring Plan

## Status

**Prepared, not activated.** No public Cloud Run service URL exists yet, so an
external check cannot be created or truthfully verified. Monitoring is the
first post-deployment action.

## Selected monitor

Use a Google Cloud Monitoring **public HTTPS uptime check** for the Cloud Run
service's unauthenticated `/health` endpoint.

- Endpoint: `https://PUBLIC_CLOUD_RUN_HOST/health`
- Method: `GET`
- Interval: 5 minutes
- Timeout: 10 seconds
- Expected HTTP status: `200`
- Optional content match: `"status":"ok"`
- Authentication: none
- Metrics to inspect: availability percentage and response latency

This deliberately checks only process availability. It does not call
authenticated inference or paid LLM endpoints, does not store an API key in
the monitor, and cannot consume model/provider quota.

Google documents that uptime checks periodically query HTTP/HTTPS endpoints,
record latency, and support alerting policies:

- <https://docs.cloud.google.com/monitoring/uptime-checks>
- <https://cloud.google.com/monitoring/uptime-checks/uptime-alerting-policies>

## Alert policy

After deployment:

1. Create an email notification channel in Cloud Monitoring.
2. Create the public HTTPS uptime check above.
3. Attach an alert policy that opens an incident after two consecutive failed
   checks and closes it after the check recovers.
4. Send the alert to the project owner's email channel.
5. Trigger one controlled validation by temporarily using an invalid path, then
   restore `/health` and confirm both firing and recovery notifications.

Two failed five-minute checks reduce noise from a single transient cold start,
but mean detection can take approximately ten minutes. This is appropriate for
a low-cost portfolio service, not for a strict production SLA.

## Cold starts and platform behavior

Cloud Run is configured with `min-instances=0` to control cost, so the first
request after scale-to-zero can be slower. The uptime chart should therefore be
used to observe both availability and latency spikes. A five-minute monitor may
also keep a lightly used instance warmer than purely organic traffic; its
latency is not a clean measure of an entirely idle cold start.

## Application logs

Use Cloud Run **Logs** or Logs Explorer, filtering by the service and the JSON
`event` field. Useful fields include:

- `request_id` and `trace_id`
- `route`, `status_code`, and `elapsed_ms`
- `agent_implementation`
- `tool_names` and `tool_call_count`
- `error_category` and `retry_count`

Queries and credentials are not logged verbatim. Search by request or trace ID
when connecting one API request to agent and tool events.

## Activation blocker

The following user-owned state is required:

- a Google Cloud project with billing enabled;
- authenticated `gcloud` access;
- the deployed Cloud Run URL;
- permission to create Monitoring checks and alert policies;
- a verified notification channel.

Until those exist, uptime, downtime, public response time, and alert delivery
remain **not measured**.
