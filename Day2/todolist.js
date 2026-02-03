/**
 * Day2 ToDoアプリ - JavaScript
 * 構成：データ・ストレージ・タイマー・期限・お気に入り・リスト・タスク操作・カウンター・描画・初期化
 */

// ===== データ =====
// タスク一覧。各要素: { id, title, isCompleted, isFavorite, dueDate, listId, memo, elapsedSeconds }
// elapsedSeconds は表示用のみ（localStorage には保存しない）
var tasks = [];
var nextId = 1;
var STORAGE_KEY = "todo_app_tasks";
var STORAGE_KEY_LISTS = "todo_app_lists";
// リスト（カテゴリ）。各要素: { listId, listName }
var lists = [];
var nextListId = 1;
var currentListId = 1;
var DEFAULT_LIST_ID = 1;

// 現在動いているタイマーのタスクID（同時に1つだけ）
var runningTimerTaskId = null;
var timerStartTime = null;
var timerIntervalId = null;

// 経過秒数を mm:ss 形式で返す
function formatElapsed(seconds) {
  var m = Math.floor(seconds / 60);
  var s = seconds % 60;
  return ("0" + m).slice(-2) + ":" + ("0" + s).slice(-2);
}

// ===== ストレージ（リスト） =====
function loadLists() {
  try {
    var json = localStorage.getItem(STORAGE_KEY_LISTS);
    if (json) {
      lists = JSON.parse(json);
      var maxLid = 0;
      for (var i = 0; i < lists.length; i++) {
        if (lists[i].listId > maxLid) maxLid = lists[i].listId;
      }
      nextListId = maxLid + 1;
    } else {
      lists = [{ listId: DEFAULT_LIST_ID, listName: "デフォルトリスト" }];
      nextListId = DEFAULT_LIST_ID + 1;
    }
  } catch (e) {
    lists = [{ listId: DEFAULT_LIST_ID, listName: "デフォルトリスト" }];
    nextListId = DEFAULT_LIST_ID + 1;
  }
}

function saveLists() {
  localStorage.setItem(STORAGE_KEY_LISTS, JSON.stringify(lists));
}

// ===== ストレージ（タスク） =====
function loadTasks() {
  try {
    var json = localStorage.getItem(STORAGE_KEY);
    if (json) {
      var saved = JSON.parse(json);
      tasks = saved.map(function (t) {
        return {
          id: t.id,
          title: t.title,
          isCompleted: t.isCompleted === true,
          isFavorite: t.isFavorite === true,
          dueDate: t.dueDate || null,
          listId: t.listId != null ? t.listId : DEFAULT_LIST_ID,
          memo: t.memo || "",
          elapsedSeconds: 0
        };
      });
      var maxId = 0;
      for (var i = 0; i < tasks.length; i++) {
        if (tasks[i].id > maxId) maxId = tasks[i].id;
      }
      nextId = maxId + 1;
    }
  } catch (e) {
    tasks = [];
  }
}

function saveTasks() {
  var toSave = tasks.map(function (t) {
    return {
      id: t.id,
      title: t.title,
      isCompleted: t.isCompleted,
      isFavorite: t.isFavorite,
      dueDate: t.dueDate,
      listId: t.listId,
      memo: t.memo
    };
  });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
}

// ===== タイマー =====
function getTaskById(id) {
  for (var i = 0; i < tasks.length; i++) {
    if (tasks[i].id === id) return tasks[i];
  }
  return null;
}

function updateTaskTimerDisplay(taskId) {
  var task = getTaskById(taskId);
  if (!task) return;
  var displayEl = document.getElementById("timer-display-" + taskId);
  if (!displayEl) return;
  var seconds;
  if (runningTimerTaskId === taskId && timerStartTime !== null) {
    seconds = task.elapsedSeconds + Math.floor((Date.now() - timerStartTime) / 1000);
  } else {
    seconds = task.elapsedSeconds;
  }
  displayEl.textContent = formatElapsed(seconds);
}

function updateAllTimerDisplays() {
  if (runningTimerTaskId !== null) {
    updateTaskTimerDisplay(runningTimerTaskId);
  }
}

function stopAnyRunningTimer() {
  if (timerIntervalId !== null) {
    clearInterval(timerIntervalId);
    timerIntervalId = null;
  }
  if (runningTimerTaskId !== null && timerStartTime !== null) {
    var task = getTaskById(runningTimerTaskId);
    if (task) {
      task.elapsedSeconds += Math.floor((Date.now() - timerStartTime) / 1000);
    }
    runningTimerTaskId = null;
    timerStartTime = null;
  }
}

function startTimer(taskId) {
  stopAnyRunningTimer();
  var task = getTaskById(taskId);
  if (!task) return;
  runningTimerTaskId = taskId;
  timerStartTime = Date.now();
  updateTaskTimerDisplay(taskId);
  timerIntervalId = setInterval(updateAllTimerDisplays, 1000);
}

function stopTimer(taskId) {
  if (runningTimerTaskId !== taskId) return;
  if (timerIntervalId !== null) {
    clearInterval(timerIntervalId);
    timerIntervalId = null;
  }
  var task = getTaskById(taskId);
  if (task && timerStartTime !== null) {
    task.elapsedSeconds += Math.floor((Date.now() - timerStartTime) / 1000);
  }
  runningTimerTaskId = null;
  timerStartTime = null;
  updateTaskTimerDisplay(taskId);
}

function resetTimer(taskId) {
  if (runningTimerTaskId === taskId) {
    stopTimer(taskId);
  }
  var task = getTaskById(taskId);
  if (task) {
    task.elapsedSeconds = 0;
    updateTaskTimerDisplay(taskId);
  }
}

// ===== 期限（締切）判定 =====
function getDueDateState(dueDate) {
  if (dueDate === null || dueDate === "") return "none";
  var today = new Date();
  var y = today.getFullYear();
  var m = ("0" + (today.getMonth() + 1)).slice(-2);
  var d = ("0" + today.getDate()).slice(-2);
  var todayStr = y + "-" + m + "-" + d;
  if (dueDate < todayStr) return "overdue";
  return "ok";
}

// ===== お気に入り =====
function toggleFavorite(taskId) {
  var task = getTaskById(taskId);
  if (task) {
    task.isFavorite = !task.isFavorite;
    saveTasks();
    renderTasks();
  }
}

// ===== 期限設定 =====
function setTaskDueDate(taskId, dateStringOrNull) {
  var task = getTaskById(taskId);
  if (task) {
    task.dueDate = dateStringOrNull || null;
    saveTasks();
    renderTasks();
  }
}

// ===== リスト操作 =====
function addList(listName) {
  var name = (listName || "").trim();
  if (name === "") return;
  lists.push({ listId: nextListId, listName: name });
  nextListId += 1;
  saveLists();
  currentListId = lists[lists.length - 1].listId;
  renderListSelect();
  renderTasks();
}

function deleteList(listId) {
  if (listId === DEFAULT_LIST_ID) return;
  for (var i = 0; i < lists.length; i++) {
    if (lists[i].listId === listId) {
      lists.splice(i, 1);
      break;
    }
  }
  for (var j = 0; j < tasks.length; j++) {
    if (tasks[j].listId === listId) tasks[j].listId = DEFAULT_LIST_ID;
  }
  saveLists();
  saveTasks();
  if (currentListId === listId) currentListId = DEFAULT_LIST_ID;
  renderListSelect();
  renderTasks();
}

function setCurrentList(listId) {
  currentListId = parseInt(listId, 10);
  renderTasks();
}

function renderListSelect() {
  var select = document.getElementById("listSelect");
  select.innerHTML = "";
  for (var i = 0; i < lists.length; i++) {
    var opt = document.createElement("option");
    opt.value = lists[i].listId;
    opt.textContent = lists[i].listName;
    if (lists[i].listId === currentListId) opt.selected = true;
    select.appendChild(opt);
  }
}

// ===== タスク操作 =====
function addTask() {
  var input = document.getElementById("taskInput");
  var title = input.value.trim();
  if (title === "") return;
  var newTask = {
    id: nextId,
    title: title,
    isCompleted: false,
    isFavorite: false,
    dueDate: null,
    listId: currentListId,
    memo: "",
    elapsedSeconds: 0
  };
  tasks.unshift(newTask);
  nextId += 1;
  input.value = "";
  saveTasks();
  renderTasks();
}

function removeTask(taskId) {
  if (runningTimerTaskId === taskId) stopAnyRunningTimer();
  for (var i = 0; i < tasks.length; i++) {
    if (tasks[i].id === taskId) {
      tasks.splice(i, 1);
      break;
    }
  }
  saveTasks();
  renderTasks();
}

function toggleCompleted(taskId) {
  var task = getTaskById(taskId);
  if (task) {
    task.isCompleted = !task.isCompleted;
    saveTasks();
    renderTasks();
  }
}

function setTaskTitle(taskId, newTitle) {
  var task = getTaskById(taskId);
  if (task) {
    task.title = newTitle.trim() || task.title;
    saveTasks();
    renderTasks();
  }
}

function setTaskMemo(taskId, memo) {
  var task = getTaskById(taskId);
  if (task) {
    task.memo = memo;
    saveTasks();
  }
}

// ===== カウンター =====
function getTaskCounts() {
  var incomplete = tasks.filter(function (t) { return !t.isCompleted; }).length;
  var completed = tasks.filter(function (t) { return t.isCompleted; }).length;
  var favorite = tasks.filter(function (t) { return t.isFavorite; }).length;
  var overdue = tasks.filter(function (t) { return getDueDateState(t.dueDate) === "overdue"; }).length;
  return { incomplete: incomplete, completed: completed, favorite: favorite, overdue: overdue };
}

function updateCounterDisplay() {
  var counts = getTaskCounts();
  document.getElementById("counterIncomplete").textContent = "未完了：" + counts.incomplete + "件";
  document.getElementById("counterCompleted").textContent = "完了：" + counts.completed + "件";
  document.getElementById("counterFavorite").textContent = "お気に入り：" + counts.favorite + "件";
  document.getElementById("counterOverdue").textContent = "期限切れ：" + counts.overdue + "件";
}

// ===== 描画 =====
function getTasksForCurrentList() {
  return tasks.filter(function (t) { return t.listId === currentListId; });
}

function renderTasks() {
  var listEl = document.getElementById("taskList");
  listEl.innerHTML = "";
  var displayTasks = getTasksForCurrentList();
  for (var i = 0; i < displayTasks.length; i++) {
    var task = displayTasks[i];
    var dueState = getDueDateState(task.dueDate);
    var card = document.createElement("li");
    card.className = "task-card" + (task.isCompleted ? " completed" : "") +
      (dueState === "overdue" ? " overdue" : "") + (task.isFavorite ? " favorite" : "");
    card.dataset.taskId = task.id;

    var header = document.createElement("div");
    header.className = "task-header";
    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = task.isCompleted;
    cb.addEventListener("change", function (id) { return function () { toggleCompleted(id); }; }(task.id));
    var titleSpan = document.createElement("span");
    titleSpan.className = "task-title";
    titleSpan.textContent = task.title;
    titleSpan.contentEditable = "false";
    var editBtn = document.createElement("button");
    editBtn.className = "btn-edit";
    editBtn.textContent = "編集";
    editBtn.addEventListener("click", function (id, span) {
      return function () {
        if (span.classList.contains("edit-mode")) {
          setTaskTitle(id, span.textContent);
          span.classList.remove("edit-mode");
          span.contentEditable = "false";
        } else {
          span.classList.add("edit-mode");
          span.contentEditable = "true";
          span.focus();
          function onBlur() {
            setTaskTitle(id, span.textContent);
            span.classList.remove("edit-mode");
            span.contentEditable = "false";
            span.removeEventListener("blur", onBlur);
          }
          span.addEventListener("blur", onBlur);
        }
      };
    }(task.id, titleSpan));
    var favBtn = document.createElement("button");
    favBtn.type = "button";
    favBtn.className = "btn-favorite" + (task.isFavorite ? " active" : "");
    favBtn.textContent = task.isFavorite ? "★" : "☆";
    favBtn.title = "お気に入り";
    favBtn.addEventListener("click", function (id) { return function () { toggleFavorite(id); }; }(task.id));
    var deleteBtn = document.createElement("button");
    deleteBtn.className = "btn-delete";
    deleteBtn.textContent = "削除";
    deleteBtn.addEventListener("click", function (id) { return function () { removeTask(id); }; }(task.id));
    header.appendChild(cb);
    header.appendChild(titleSpan);
    header.appendChild(favBtn);
    header.appendChild(editBtn);
    header.appendChild(deleteBtn);
    card.appendChild(header);

    var dueRow = document.createElement("div");
    dueRow.className = "due-row";
    var dueInput = document.createElement("input");
    dueInput.type = "date";
    dueInput.value = task.dueDate || "";
    dueInput.addEventListener("change", function (id) {
      return function () { setTaskDueDate(id, this.value || null); };
    }(task.id));
    var dueStateSpan = document.createElement("span");
    dueStateSpan.className = "due-state";
    dueStateSpan.textContent = dueState === "none" ? "期限なし" : dueState === "ok" ? "期限内" : "期限切れ";
    dueRow.appendChild(dueInput);
    dueRow.appendChild(dueStateSpan);
    card.appendChild(dueRow);

    var timerRow = document.createElement("div");
    timerRow.className = "timer-row";
    timerRow.innerHTML = "<span class=\"timer-label\">タイマー：</span><span class=\"timer-display\" id=\"timer-display-" + task.id + "\">" + formatElapsed(task.elapsedSeconds) + "</span>";
    var timerBtns = document.createElement("div");
    timerBtns.className = "timer-btns";
    var startBtn = document.createElement("button");
    startBtn.className = "timer-start";
    startBtn.textContent = "開始";
    startBtn.addEventListener("click", function (id) { return function () { startTimer(id); }; }(task.id));
    var stopBtn = document.createElement("button");
    stopBtn.className = "timer-stop";
    stopBtn.textContent = "停止";
    stopBtn.addEventListener("click", function (id) { return function () { stopTimer(id); }; }(task.id));
    var resetBtn = document.createElement("button");
    resetBtn.className = "timer-reset";
    resetBtn.textContent = "リセット";
    resetBtn.addEventListener("click", function (id) { return function () { resetTimer(id); }; }(task.id));
    timerBtns.appendChild(startBtn);
    timerBtns.appendChild(stopBtn);
    timerBtns.appendChild(resetBtn);
    timerRow.appendChild(timerBtns);
    card.appendChild(timerRow);

    var memoLabel = document.createElement("label");
    memoLabel.className = "memo-label";
    memoLabel.textContent = "メモ：";
    var memoArea = document.createElement("textarea");
    memoArea.className = "memo-textarea";
    memoArea.placeholder = "メモを入力";
    memoArea.value = task.memo;
    memoArea.addEventListener("input", function (id) {
      return function () { setTaskMemo(id, this.value); };
    }(task.id));
    card.appendChild(memoLabel);
    card.appendChild(memoArea);

    listEl.appendChild(card);
  }
  updateCounterDisplay();
}

// ===== 初期化 =====
document.getElementById("addButton").addEventListener("click", addTask);
document.getElementById("taskInput").addEventListener("keydown", function (e) {
  if (e.key === "Enter") addTask();
});
document.getElementById("listSelect").addEventListener("change", function () {
  setCurrentList(parseInt(this.value, 10));
});
document.getElementById("addListBtn").addEventListener("click", function () {
  var name = window.prompt("新しいリスト名を入力してください");
  if (name !== null) addList(name);
});
document.getElementById("deleteListBtn").addEventListener("click", function () {
  if (currentListId === DEFAULT_LIST_ID) {
    window.alert("デフォルトリストは削除できません");
    return;
  }
  if (window.confirm("このリストを削除しますか？\n属するタスクはデフォルトリストに移動します。")) {
    deleteList(currentListId);
  }
});

loadLists();
loadTasks();
renderListSelect();
renderTasks();
