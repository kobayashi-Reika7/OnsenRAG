/**
 * バックエンド（FastAPI）への fetch 呼び出し
 * ベースURL: 開発時は localhost:8000
 */
const API_BASE = 'http://localhost:8000';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  if (res.status === 204) return;
  return res.json();
}

// ===== タスク API =====
export async function getTasks() {
  return request('/tasks');
}

export async function createTask(data) {
  return request('/tasks', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateTask(id, data) {
  return request(`/tasks/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteTask(id) {
  return request(`/tasks/${id}`, { method: 'DELETE' });
}

// ===== リスト API =====
export async function getLists() {
  return request('/lists');
}

export async function createList(name) {
  return request('/lists', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function deleteList(id) {
  return request(`/lists/${id}`, { method: 'DELETE' });
}
