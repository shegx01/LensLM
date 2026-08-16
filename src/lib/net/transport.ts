/**
 * Client-side transport gate for any cloud endpoint an API key is bearer-sent to.
 * Cleartext http: is confined to loopback and private/LAN hosts; on a LAN the key still
 * crosses the network in the clear, which is the trade this rule makes deliberately.
 * SYNC-CHECK: mirrors `is_transport_safe_base_url` in `lens-core/src/http.rs`.
 */
function isPrivateNetworkHost(hostname: string): boolean {
  const host = hostname.toLowerCase();
  if (host === 'localhost' || host === '[::1]' || host.endsWith('.local')) return true;
  const octets = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!octets) return false;
  const a = Number(octets[1]);
  const b = Number(octets[2]);
  return a === 127 || a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

export function isTransportSafeBaseUrl(raw: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  if (parsed.protocol === 'https:') return true;
  return parsed.protocol === 'http:' && isPrivateNetworkHost(parsed.hostname);
}
