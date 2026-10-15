/*
 * Starting Chrome, with the flakiness separated from the test.
 *
 * On GitHub's runners the launch itself occasionally fails -- "Timed out
 * waiting for the WS endpoint" -- while the browser and the page are both
 * fine. That is an infrastructure problem, not a result, and it used to go
 * red and block the Pages deployment for a reason no commit could fix.
 *
 * So: retry a few times, and if it still will not start, say so in a way the
 * caller can tell apart from a page that misbehaved. A launch that never
 * happened is reported as a skip; anything the page did once it loaded is
 * still a failure.
 */
import puppeteer from "puppeteer-core";

export const LAUNCH_FAILED = "launchFailed";

const ATTEMPTS = 3;
const BACKOFF_MS = 2000;

export async function launchChrome(chromePath) {
  const failures = [];
  for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
    try {
      return await puppeteer.launch({
        executablePath: chromePath,
        headless: "new",
        args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        timeout: 60000,
      });
    } catch (error) {
      failures.push(`attempt ${attempt}: ${error.message}`);
      if (attempt < ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, BACKOFF_MS * attempt));
      }
    }
  }
  const error = new Error(failures.join("; "));
  error[LAUNCH_FAILED] = true;
  throw error;
}

/*
 * Print the report and exit. A browser that never started exits 0 with
 * launchFailed set, so the caller skips rather than fails.
 */
export function reportAndExit(report, error) {
  if (error && error[LAUNCH_FAILED]) {
    report.ok = false;
    report.launchFailed = true;
    report.failure = `Chrome would not start: ${error.message}`;
  } else if (error) {
    report.ok = false;
    report.failure = error.message;
  }
  process.stdout.write(JSON.stringify(report, null, 1));
  process.exit(0);
}
