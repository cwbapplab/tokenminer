/**
 * The device's stable identity. The server derives the pool-facing worker id from this,
 * so it must be generated once and never change for the lifetime of the install.
 */
const HARDWARE_ID_KEY = "tokenminer.hardwareId";

export function getHardwareId(): string {
  let id = localStorage.getItem(HARDWARE_ID_KEY);

  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(HARDWARE_ID_KEY, id);
  }

  return id;
}
