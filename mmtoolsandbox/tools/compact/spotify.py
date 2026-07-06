# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Spotify tools — playback, queue, volume, search, show, liked, library, privates, subscription."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.spotify as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Playback (7 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_playback(
    action: Literal["play", "pause", "next", "previous", "seek", "loop", "shuffle"],
    song_id: int | None = None,
    album_id: int | None = None,
    playlist_id: int | None = None,
    queue_position: int | None = None,
    seek_seconds: int | NotGiven = NOT_GIVEN,
    loop: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Play, pause, skip, seek, loop, or shuffle music on Spotify.

    Actions:
        play: Play music. Optionally pass one of song_id, album_id,
            playlist_id, or queue_position. If song_id, album_id, or
            playlist_id is passed, it will be added to the queue and played.
            If queue_position is passed, the song at that position is played.
            If none is passed, the current song is played.
        pause: Pause the currently playing song.
        next: Go to the next song in the queue.
        previous: Go to the previous song in the queue.
        seek: Seek the current song. Requires seek_seconds.
        loop: Set whether to loop the current song. Requires loop.
        shuffle: Shuffle songs in the queue.

    Args:
        action: The playback operation to perform.
        song_id: ID of the song to play (for play).
        album_id: ID of the album to play (for play).
        playlist_id: ID of the playlist to play (for play).
        queue_position: Position in queue to play (for play).
        seek_seconds: Number of seconds to seek to (for seek).
        loop: Whether to loop the current song (for loop).

    Returns:
        Playback status or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "play":
        kwargs: dict[str, Any] = {}
        if song_id is not None:
            kwargs["song_id"] = song_id
        if album_id is not None:
            kwargs["album_id"] = album_id
        if playlist_id is not None:
            kwargs["playlist_id"] = playlist_id
        if queue_position is not None:
            kwargs["queue_position"] = queue_position
        return _get("spotify_play_music")(**kwargs)
    elif action == "pause":
        return _get("spotify_pause_music")()
    elif action == "next":
        return _get("spotify_next_song")()
    elif action == "previous":
        return _get("spotify_previous_song")()
    elif action == "seek":
        return _get("spotify_seek_song")(seek_seconds=seek_seconds)
    elif action == "loop":
        return _get("spotify_loop_song")(loop=loop)
    elif action == "shuffle":
        return _get("spotify_shuffle_song_queue")()
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 5: Queue management (5 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_queue(
    action: Literal["show", "add", "remove", "clear", "move"],
    song_id: int | None = None,
    album_id: int | None = None,
    playlist_id: int | None = None,
    position: int | NotGiven = NOT_GIVEN,
    current_position: int | NotGiven = NOT_GIVEN,
    new_position: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show, add to, remove from, clear, or reorder the Spotify song queue.

    Actions:
        show: Show the current song queue.
        add: Add a song, album, or playlist to the queue. Optionally pass
            one of song_id, album_id, or playlist_id.
        remove: Remove a song from the queue. Requires position.
        clear: Clear the entire song queue.
        move: Move a song in the queue. Requires current_position and
            new_position.

    Args:
        action: The queue operation to perform.
        song_id: ID of the song to add (for add).
        album_id: ID of the album to add (for add).
        playlist_id: ID of the playlist to add (for add).
        position: 0-indexed position of the song to remove (for remove).
        current_position: Current position of the song to move (for move).
        new_position: New position for the song (for move).

    Returns:
        Queue contents or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "show":
        return _get("spotify_show_song_queue")()
    elif action == "add":
        kwargs: dict[str, Any] = {}
        if song_id is not None:
            kwargs["song_id"] = song_id
        if album_id is not None:
            kwargs["album_id"] = album_id
        if playlist_id is not None:
            kwargs["playlist_id"] = playlist_id
        return _get("spotify_add_to_queue")(**kwargs)
    elif action == "remove":
        return _get("spotify_remove_song_from_queue")(position=position)
    elif action == "clear":
        return _get("spotify_clear_song_queue")()
    elif action == "move":
        return _get("spotify_move_song_in_queue")(
            current_position=current_position, new_position=new_position
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 5: Volume management (2 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_volume(
    action: Literal["show", "set"],
    volume: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Show or set the Spotify music player volume level.

    Actions:
        show: Get the current volume level.
        set: Set the volume level. Requires volume.

    Args:
        action: The volume operation to perform.
        volume: Volume level to set (for set).

    Returns:
        Current or updated volume details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "show":
        return _get("spotify_show_volume")()
    elif action == "set":
        return _get("spotify_set_volume")(volume=volume)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 4: Show entity (3 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_show_entity(
    entity_type: Literal["song", "album", "artist"],
    entity_id: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Show details of a Spotify song, album, or artist by ID.

    Args:
        entity_type: The type of entity to show.
        entity_id: The ID of the entity to retrieve.

    Returns:
        Entity details.

    Raises:
        ConnectionError: If network is unavailable.
    """
    func_map: dict[str, tuple[str, str]] = {
        "song": ("spotify_show_song", "song_id"),
        "album": ("spotify_show_album", "album_id"),
        "artist": ("spotify_show_artist", "artist_id"),
    }
    func_name, param_name = func_map[entity_type]
    return _get(func_name)(**{param_name: entity_id})


# ---------------------------------------------------------------------------
# Strategy 4: Show liked (3 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_show_liked(
    entity_type: Literal["song", "album", "playlist"],
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = "-liked_at",
) -> list[dict[str, Any]]:
    """Show liked songs, albums, or playlists on Spotify.

    Args:
        entity_type: The type of liked entity to list.
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field. Prefix with '-' for descending.
            Valid for song: liked_at, play_count, title.
            Valid for album/playlist: liked_at, title.

    Returns:
        List of liked entities.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    func_map = {
        "song": "spotify_show_liked_songs",
        "album": "spotify_show_liked_albums",
        "playlist": "spotify_show_liked_playlists",
    }
    kwargs: dict[str, Any] = {}
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit
    if sort_by is not None:
        kwargs["sort_by"] = sort_by
    return _get(func_map[entity_type])(**kwargs)


# ---------------------------------------------------------------------------
# Strategy 4: Show library (2 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_show_library(
    entity_type: Literal["song", "album"],
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """Search or show songs or albums in the Spotify library.

    Args:
        entity_type: The type of library entity to list.
        query: Search query string.
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field. Prefix with '-' for descending.
            Valid attributes: added_at, title. Defaults to -added_at
            when both query and sort_by are empty.

    Returns:
        List of library entities.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    func_map = {
        "song": "spotify_show_song_library",
        "album": "spotify_show_album_library",
    }
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit
    if sort_by is not None:
        kwargs["sort_by"] = sort_by
    return _get(func_map[entity_type])(**kwargs)


# ---------------------------------------------------------------------------
# Strategy 4: Show privates (3 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_show_privates(
    entity_type: Literal["song", "album", "playlist"],
    entity_id: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Show private user-specific information about a Spotify song, album, or playlist.

    Args:
        entity_type: The type of entity.
        entity_id: The ID of the entity.

    Returns:
        Private entity details for the current user.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    func_map: dict[str, tuple[str, str]] = {
        "song": ("spotify_show_song_privates", "song_id"),
        "album": ("spotify_show_album_privates", "album_id"),
        "playlist": ("spotify_show_playlist_privates", "playlist_id"),
    }
    func_name, param_name = func_map[entity_type]
    return _get(func_name)(**{param_name: entity_id})


# ---------------------------------------------------------------------------
# Strategy 4: Search (4 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_search(
    entity_type: Literal["song", "album", "playlist", "artist"],
    query: str | None = "",
    # Song-specific
    artist_id: int | None = None,
    album_id: int | None = None,
    # Song/album/artist shared
    genre: str | None = None,
    # Song/album shared
    min_release_date: str | None = None,
    max_release_date: str | None = None,
    # Song-only
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_play_count: int | None = None,
    max_play_count: int | None = None,
    # Song/album/playlist shared
    min_rating: float | None = None,
    max_rating: float | None = None,
    min_like_count: int | None = None,
    max_like_count: int | None = None,
    # Playlist-specific
    owner_email: str | None = None,
    # Artist-specific
    min_follower_count: int | None = None,
    max_follower_count: int | None = None,
    # Pagination
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """Search for songs, albums, playlists, or artists on Spotify.

    Args:
        entity_type: The type of entity to search for.
        query: Search query string.
        artist_id: Filter songs by artist ID (song only).
        album_id: Filter songs by album ID (song only).
        genre: Filter by genre (song, album, artist).
        min_release_date: Minimum release date, format YYYY-MM-DD (song, album).
        max_release_date: Maximum release date, format YYYY-MM-DD (song, album).
        min_duration: Minimum duration in seconds (song only).
        max_duration: Maximum duration in seconds (song only).
        min_play_count: Minimum play count (song only).
        max_play_count: Maximum play count (song only).
        min_rating: Minimum rating (song, album, playlist).
        max_rating: Maximum rating (song, album, playlist).
        min_like_count: Minimum like count (song, album, playlist).
        max_like_count: Maximum like count (song, album, playlist).
        owner_email: Filter playlists by owner email (playlist only).
        min_follower_count: Minimum follower count (artist only).
        max_follower_count: Maximum follower count (artist only).
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field. Prefix with '-' for descending.

    Returns:
        List of matching entities.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    # Shared pagination params
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit
    if sort_by is not None:
        kwargs["sort_by"] = sort_by

    if entity_type == "song":
        if artist_id is not None:
            kwargs["artist_id"] = artist_id
        if album_id is not None:
            kwargs["album_id"] = album_id
        if genre is not None:
            kwargs["genre"] = genre
        if min_release_date is not None:
            kwargs["min_release_date"] = min_release_date
        if max_release_date is not None:
            kwargs["max_release_date"] = max_release_date
        if min_duration is not None:
            kwargs["min_duration"] = min_duration
        if max_duration is not None:
            kwargs["max_duration"] = max_duration
        if min_play_count is not None:
            kwargs["min_play_count"] = min_play_count
        if max_play_count is not None:
            kwargs["max_play_count"] = max_play_count
        if min_rating is not None:
            kwargs["min_rating"] = min_rating
        if max_rating is not None:
            kwargs["max_rating"] = max_rating
        if min_like_count is not None:
            kwargs["min_like_count"] = min_like_count
        if max_like_count is not None:
            kwargs["max_like_count"] = max_like_count
        return _get("spotify_search_songs")(**kwargs)
    elif entity_type == "album":
        if genre is not None:
            kwargs["genre"] = genre
        if min_release_date is not None:
            kwargs["min_release_date"] = min_release_date
        if max_release_date is not None:
            kwargs["max_release_date"] = max_release_date
        if min_rating is not None:
            kwargs["min_rating"] = min_rating
        if max_rating is not None:
            kwargs["max_rating"] = max_rating
        if min_like_count is not None:
            kwargs["min_like_count"] = min_like_count
        if max_like_count is not None:
            kwargs["max_like_count"] = max_like_count
        return _get("spotify_search_albums")(**kwargs)
    elif entity_type == "playlist":
        if min_rating is not None:
            kwargs["min_rating"] = min_rating
        if max_rating is not None:
            kwargs["max_rating"] = max_rating
        if min_like_count is not None:
            kwargs["min_like_count"] = min_like_count
        if max_like_count is not None:
            kwargs["max_like_count"] = max_like_count
        if owner_email is not None:
            kwargs["owner_email"] = owner_email
        return _get("spotify_search_playlists")(**kwargs)
    elif entity_type == "artist":
        if genre is not None:
            kwargs["genre"] = genre
        if min_follower_count is not None:
            kwargs["min_follower_count"] = min_follower_count
        if max_follower_count is not None:
            kwargs["max_follower_count"] = max_follower_count
        return _get("spotify_search_artists")(**kwargs)
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")


# ---------------------------------------------------------------------------
# Absorption declarations
# ---------------------------------------------------------------------------

mark_compact_tools_absorbed_by(
    "spotify_playback",
    "spotify_play_music",
    "spotify_pause_music",
    "spotify_next_song",
    "spotify_previous_song",
    "spotify_seek_song",
    "spotify_loop_song",
    "spotify_shuffle_song_queue",
)
mark_compact_tools_absorbed_by(
    "spotify_manage_queue",
    "spotify_show_song_queue",
    "spotify_add_to_queue",
    "spotify_remove_song_from_queue",
    "spotify_clear_song_queue",
    "spotify_move_song_in_queue",
)
mark_compact_tools_absorbed_by(
    "spotify_manage_volume",
    "spotify_show_volume",
    "spotify_set_volume",
)
mark_compact_tools_absorbed_by(
    "spotify_show_entity",
    "spotify_show_song",
    "spotify_show_album",
    "spotify_show_artist",
)
mark_compact_tools_absorbed_by(
    "spotify_show_liked",
    "spotify_show_liked_songs",
    "spotify_show_liked_albums",
    "spotify_show_liked_playlists",
)
mark_compact_tools_absorbed_by(
    "spotify_show_library",
    "spotify_show_song_library",
    "spotify_show_album_library",
)
mark_compact_tools_absorbed_by(
    "spotify_show_privates",
    "spotify_show_song_privates",
    "spotify_show_album_privates",
    "spotify_show_playlist_privates",
)
mark_compact_tools_absorbed_by(
    "spotify_search",
    "spotify_search_songs",
    "spotify_search_albums",
    "spotify_search_playlists",
    "spotify_search_artists",
)


# ---------------------------------------------------------------------------
# Lazy import helper — dispatch to MEDIUM consolidated Spotify tools
# ---------------------------------------------------------------------------


def _get_consolidated(name: str) -> Any:
    import mmtoolsandbox.tools.consolidated.spotify as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD+List: Spotify playlist management (absorbs spotify_show_playlist_library)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_playlist(
    action: Literal["create", "show", "update", "delete", "list"],
    playlist_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    is_public: bool | NotGiven = NOT_GIVEN,
    # list-action params
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Spotify playlists: create, view, update, delete, or list.

    Actions:
        create: Create a new playlist. Requires title. is_public defaults
            to false.
        show: View details of a playlist. Requires playlist_id.
        update: Update title or privacy of a playlist. Requires playlist_id
            and at least one of title or is_public.
        delete: Delete a playlist. Requires playlist_id.
        list: List playlists in your playlist library. Supports query,
            is_public filter, pagination, and sorting.

    Args:
        action: The operation to perform.
        playlist_id: The playlist ID (for show, update, delete).
        title: Playlist title (for create, update).
        is_public: Whether the playlist is public (for create, update,
            list filter).
        query: Search query string (for list).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: created_at, title (for list).

    Returns:
        Playlist details, action confirmation, or list of playlists.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"title": title}
        if is_public is not NOT_GIVEN:
            kwargs["is_public"] = is_public
        return _get("spotify_create_playlist")(**kwargs)
    elif action == "show":
        return _get("spotify_show_playlist")(playlist_id=playlist_id)
    elif action == "update":
        kwargs = {"playlist_id": playlist_id}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if is_public is not NOT_GIVEN:
            kwargs["is_public"] = is_public
        return _get("spotify_update_playlist")(**kwargs)
    elif action == "delete":
        return _get("spotify_delete_playlist")(playlist_id=playlist_id)
    elif action == "list":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if is_public is not NOT_GIVEN and is_public is not None:
            kwargs["is_public"] = is_public
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("spotify_show_playlist_library")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "spotify_manage_playlist",
    "spotify_show_playlist_library",
)


# ---------------------------------------------------------------------------
# Collapse MEDIUM: spotify_toggle_like — replaces 3 MEDIUM tools
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_toggle_like(
    entity_type: Literal["song", "album", "playlist"],
    entity_id: int,
    like: bool,
) -> dict[str, Any]:
    """Like or unlike a song, album, or playlist on Spotify.

    Args:
        entity_type: The type of entity to like or unlike.
        entity_id: The ID of the entity.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    func_map: dict[str, tuple[str, str]] = {
        "song": ("spotify_toggle_song_like", "song_id"),
        "album": ("spotify_toggle_album_like", "album_id"),
        "playlist": ("spotify_toggle_playlist_like", "playlist_id"),
    }
    func_name, param_name = func_map[entity_type]
    return _get_consolidated(func_name)(**{param_name: entity_id, "like": like})


mark_compact_tools_absorbed_by(
    "spotify_toggle_like",
    "spotify_toggle_song_like",
    "spotify_toggle_album_like",
    "spotify_toggle_playlist_like",
)


# ---------------------------------------------------------------------------
# Collapse MEDIUM: spotify_manage_in_library — replaces 2 MEDIUM tools
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_in_library(
    entity_type: Literal["song", "album"],
    entity_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a song or album from your Spotify library.

    Args:
        entity_type: The type of entity — "song" or "album".
        entity_id: The ID of the entity.
        action: "add" to add to library, "remove" to remove from library.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    func_map: dict[str, tuple[str, str]] = {
        "song": ("spotify_manage_song_in_library", "song_id"),
        "album": ("spotify_manage_album_in_library", "album_id"),
    }
    func_name, param_name = func_map[entity_type]
    return _get_consolidated(func_name)(**{param_name: entity_id, "action": action})


mark_compact_tools_absorbed_by(
    "spotify_manage_in_library",
    "spotify_manage_song_in_library",
    "spotify_manage_album_in_library",
)
