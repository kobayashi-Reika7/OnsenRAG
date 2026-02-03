/**
 * リスト選択・追加・削除
 * 選択変更時に onSelect を呼ぶ。追加・削除は親のハンドラを呼ぶ
 */
import React from 'react';

function ListSelector({ lists, currentListId, onSelect, onAdd, onDelete }) {
  const isDefault = currentListId === 1;

  return (
    <div className="list-section">
      <label htmlFor="listSelect">リスト：</label>
      <select
        id="listSelect"
        value={currentListId}
        onChange={(e) => onSelect(Number(e.target.value))}
      >
        {lists.map((list) => (
          <option key={list.id} value={list.id}>
            {list.name}
          </option>
        ))}
      </select>
      <button type="button" className="btn-add-list" onClick={onAdd}>
        リスト追加
      </button>
      <button
        type="button"
        className="btn-delete-list"
        onClick={onDelete}
        disabled={isDefault}
      >
        リスト削除
      </button>
    </div>
  );
}

export default ListSelector;
