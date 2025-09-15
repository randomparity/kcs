import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const mcpToolRequests = new Counter('mcp_tool_requests');
const mcpToolErrors = new Rate('mcp_tool_errors');
const mcpResponseTime = new Trend('mcp_response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users over 2 minutes
    { duration: '5m', target: 10 },   // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 },   // Ramp up to 20 users over 2 minutes
    { duration: '5m', target: 20 },   // Stay at 20 users for 5 minutes
    { duration: '2m', target: 0 },    // Ramp down to 0 users over 2 minutes
  ],
  thresholds: {
    http_req_duration: ['p(95)<600'],  // Constitutional requirement: p95 < 600ms
    http_req_failed: ['rate<0.1'],     // Error rate should be less than 10%
    mcp_response_time: ['p(95)<600'],  // MCP-specific response time threshold
    mcp_tool_errors: ['rate<0.05'],    // Tool error rate should be less than 5%
  },
};

// Environment configuration
const BASE_URL = __ENV.KCS_SERVER_URL || 'http://localhost:8080';
const AUTH_TOKEN = __ENV.KCS_AUTH_TOKEN || 'test-token';

// Common headers
const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${AUTH_TOKEN}`,
};

// Test data for various endpoints
const testQueries = [
  'read from file descriptor',
  'memory allocation',
  'syscall implementation',
  'vfs operations',
  'network socket',
  'process scheduling',
  'memory management',
  'file system',
  'device driver',
  'interrupt handling'
];

const testSymbols = [
  'sys_read',
  'sys_write',
  'vfs_read',
  'vfs_write',
  'kmalloc',
  'schedule',
  'do_fork',
  'sys_open',
  'sys_close',
  'tcp_sendmsg'
];

const testEntrypoints = [
  '__NR_read',
  '__NR_write',
  '__NR_open',
  '__NR_close',
  '__NR_fork',
  '__NR_execve',
  '__NR_mmap',
  '__NR_munmap',
  '__NR_socket',
  '__NR_connect'
];

// Helper function to get random item from array
function getRandomItem(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Helper function to make MCP tool request
function mcpToolRequest(endpoint, payload) {
  const url = `${BASE_URL}/mcp/tools/${endpoint}`;
  const response = http.post(url, JSON.stringify(payload), { headers });

  mcpToolRequests.add(1);
  mcpResponseTime.add(response.timings.duration);

  const success = check(response, {
    [`${endpoint} status is 200`]: (r) => r.status === 200,
    [`${endpoint} has response body`]: (r) => r.body.length > 0,
    [`${endpoint} response time < 600ms`]: (r) => r.timings.duration < 600,
  });

  if (!success) {
    mcpToolErrors.add(1);
  }

  return response;
}

// Test scenarios for different MCP endpoints
export function testSearchCode() {
  const query = getRandomItem(testQueries);
  const payload = {
    query: query,
    topK: Math.floor(Math.random() * 20) + 5  // 5-25 results
  };

  const response = mcpToolRequest('search_code', payload);

  check(response, {
    'search_code returns hits array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.hits);
      } catch {
        return false;
      }
    }
  });
}

export function testGetSymbol() {
  const symbol = getRandomItem(testSymbols);
  const payload = { symbol: symbol };

  const response = mcpToolRequest('get_symbol', payload);

  check(response, {
    'get_symbol returns symbol info': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.name && data.kind && data.decl;
      } catch {
        return false;
      }
    }
  });
}

export function testWhoCalls() {
  const symbol = getRandomItem(testSymbols);
  const payload = {
    symbol: symbol,
    depth: Math.floor(Math.random() * 3) + 1  // 1-3 depth
  };

  const response = mcpToolRequest('who_calls', payload);

  check(response, {
    'who_calls returns callers array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.callers);
      } catch {
        return false;
      }
    }
  });
}

export function testListDependencies() {
  const symbol = getRandomItem(testSymbols);
  const payload = {
    symbol: symbol,
    depth: Math.floor(Math.random() * 2) + 1  // 1-2 depth
  };

  const response = mcpToolRequest('list_dependencies', payload);

  check(response, {
    'list_dependencies returns callees array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.callees);
      } catch {
        return false;
      }
    }
  });
}

export function testEntrypointFlow() {
  const entry = getRandomItem(testEntrypoints);
  const payload = { entry: entry };

  const response = mcpToolRequest('entrypoint_flow', payload);

  check(response, {
    'entrypoint_flow returns steps array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.steps);
      } catch {
        return false;
      }
    }
  });
}

export function testImpactOf() {
  const files = [getRandomItem(['fs/read_write.c', 'mm/mmap.c', 'net/socket.c'])];
  const symbols = [getRandomItem(testSymbols)];
  const payload = {
    files: files,
    symbols: symbols,
    config: 'x86_64:defconfig'
  };

  const response = mcpToolRequest('impact_of', payload);

  check(response, {
    'impact_of returns impact analysis': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.configs && data.modules && data.tests && data.owners;
      } catch {
        return false;
      }
    }
  });
}

export function testSearchDocs() {
  const queries = ['memory barriers', 'locking', 'atomic operations', 'rcu', 'workqueues'];
  const query = getRandomItem(queries);
  const payload = {
    query: query,
    corpus: ['Documentation/']
  };

  const response = mcpToolRequest('search_docs', payload);

  check(response, {
    'search_docs returns hits array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.hits);
      } catch {
        return false;
      }
    }
  });
}

export function testOwnersFor() {
  const paths = [getRandomItem(['fs/ext4/', 'fs/btrfs/', 'drivers/net/'])];
  const payload = { paths: paths };

  const response = mcpToolRequest('owners_for', payload);

  check(response, {
    'owners_for returns maintainers array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.maintainers);
      } catch {
        return false;
      }
    }
  });
}

// Test system endpoints
export function testHealth() {
  const response = http.get(`${BASE_URL}/health`);

  check(response, {
    'health endpoint status is 200': (r) => r.status === 200,
    'health endpoint returns status': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.status === 'healthy';
      } catch {
        return false;
      }
    }
  });
}

export function testMetrics() {
  const response = http.get(`${BASE_URL}/metrics`, { headers });

  check(response, {
    'metrics endpoint status is 200': (r) => r.status === 200,
    'metrics returns prometheus format': (r) => r.headers['Content-Type'].includes('text/plain'),
  });
}

export function testMcpResources() {
  const response = http.get(`${BASE_URL}/mcp/resources`, { headers });

  check(response, {
    'mcp resources status is 200': (r) => r.status === 200,
    'mcp resources returns array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.resources);
      } catch {
        return false;
      }
    }
  });
}

// Main test function - randomly executes different test scenarios
export default function () {
  // Randomly select which test to run to simulate realistic usage patterns
  const testFunctions = [
    testSearchCode,      // Most common - 30%
    testGetSymbol,       // Very common - 25%
    testWhoCalls,        // Common - 15%
    testListDependencies,// Common - 10%
    testEntrypointFlow,  // Medium - 8%
    testImpactOf,        // Medium - 5%
    testSearchDocs,      // Less common - 3%
    testOwnersFor,       // Less common - 2%
    testHealth,          // System - 1%
    testMetrics,         // System - 1%
  ];

  const weights = [30, 25, 15, 10, 8, 5, 3, 2, 1, 1];
  let totalWeight = weights.reduce((a, b) => a + b, 0);
  let random = Math.random() * totalWeight;

  for (let i = 0; i < weights.length; i++) {
    random -= weights[i];
    if (random <= 0) {
      testFunctions[i]();
      break;
    }
  }

  // Occasionally test MCP resources endpoint
  if (Math.random() < 0.1) {
    testMcpResources();
  }

  // Simulate realistic user think time
  sleep(Math.random() * 2 + 0.5); // 0.5-2.5 seconds
}

// Setup function - runs once at the beginning
export function setup() {
  console.log('Starting KCS MCP Load Test');
  console.log(`Target server: ${BASE_URL}`);
  console.log('Performance targets: p95 < 600ms, error rate < 10%');

  // Verify server is accessible
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`Server not accessible: ${BASE_URL}/health returned ${healthResponse.status}`);
  }

  console.log('Server health check passed');
  return { startTime: new Date() };
}

// Teardown function - runs once at the end
export function teardown(data) {
  const endTime = new Date();
  const duration = (endTime - data.startTime) / 1000;
  console.log(`Load test completed in ${duration} seconds`);
}
