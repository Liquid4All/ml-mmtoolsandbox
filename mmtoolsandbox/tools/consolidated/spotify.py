# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Spotify tools for the MEDIUM toolbox.

CRUD consolidation for playlists and reviews, plus symmetric pair merges
for like/unlike, follow/unfollow, library add/remove, and downloads.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.spotify as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Playlist management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_playlist(
    action: Literal["create", "show", "update", "delete"],
    playlist_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    is_public: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Spotify playlists: create, view, update, or delete.

    Actions:
        create: Create a new playlist. Requires title. is_public defaults
            to false.
        show: View details of a playlist. Requires playlist_id.
        update: Update title or privacy of a playlist. Requires playlist_id
            and at least one of title or is_public.
        delete: Delete a playlist. Requires playlist_id.

    Args:
        action: The operation to perform.
        playlist_id: The playlist ID (for show, update, delete).
        title: Playlist title (for create, update).
        is_public: Whether the playlist is public (for create, update).

    Returns:
        Playlist details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Cross-entity CRUD: Review management (song, album, playlist)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_review(
    entity_type: Literal["song", "album", "playlist"],
    action: Literal["list", "show", "write", "update", "delete"],
    song_id: int | NotGiven = NOT_GIVEN,
    album_id: int | NotGiven = NOT_GIVEN,
    playlist_id: int | NotGiven = NOT_GIVEN,
    review_id: int | NotGiven = NOT_GIVEN,
    rating: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    text: str | NotGiven = NOT_GIVEN,
    query: str | None = "",
    user_email: str | None = None,
    min_rating: int | None = 1,
    max_rating: int | None = 5,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage reviews for songs, albums, or playlists on Spotify.

    Actions:
        list: List reviews for an entity. Requires the entity ID
            (song_id, album_id, or playlist_id depending on entity_type).
            Supports filtering by query, user_email, rating range, and
            pagination.
        show: Show a single review by ID. Requires review_id.
        write: Write a new review. Requires the entity ID and rating.
            Optionally include title and text.
        update: Update an existing review. Requires review_id and at least
            one of rating, title, or text.
        delete: Delete a review. Requires review_id.

    Args:
        entity_type: The type of content being reviewed.
        action: The operation to perform.
        song_id: Song ID (when entity_type is "song", for list/write).
        album_id: Album ID (when entity_type is "album", for list/write).
        playlist_id: Playlist ID (when entity_type is "playlist", for
            list/write).
        review_id: Review ID (for show, update, delete).
        rating: Rating value (for write, update).
        title: Review title (for write, update).
        text: Review body text (for write, update).
        query: Search query for filtering reviews (for list).
        user_email: Filter reviews by author email (for list).
        min_rating: Minimum rating filter (for list). Defaults to 1.
        max_rating: Maximum rating filter (for list). Defaults to 5.
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: rating, created_at (for list).

    Returns:
        Review details, list of reviews, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    entity_id_map = {"song": song_id, "album": album_id, "playlist": playlist_id}
    entity_id = entity_id_map[entity_type]
    id_param = f"{entity_type}_id"

    if action == "list":
        func_name = f"spotify_show_{entity_type}_reviews"
        kwargs: dict[str, Any] = {id_param: entity_id}
        if query is not None:
            kwargs["query"] = query
        if user_email is not None:
            kwargs["user_email"] = user_email
        if min_rating is not None:
            kwargs["min_rating"] = min_rating
        if max_rating is not None:
            kwargs["max_rating"] = max_rating
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get(func_name)(**kwargs)
    elif action == "show":
        return _get(f"spotify_show_{entity_type}_review")(review_id=review_id)
    elif action == "write":
        func_name = f"spotify_review_{entity_type}"
        kwargs = {id_param: entity_id, "rating": rating}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if text is not NOT_GIVEN:
            kwargs["text"] = text
        return _get(func_name)(**kwargs)
    elif action == "update":
        func_name = f"spotify_update_{entity_type}_review"
        kwargs = {"review_id": review_id}
        if rating is not NOT_GIVEN:
            kwargs["rating"] = rating
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if text is not NOT_GIVEN:
            kwargs["text"] = text
        return _get(func_name)(**kwargs)
    elif action == "delete":
        return _get(f"spotify_delete_{entity_type}_review")(review_id=review_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Symmetric pairs: Like/Unlike
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_toggle_song_like(song_id: int, like: bool) -> dict[str, Any]:
    """Like or unlike a song on Spotify.

    Args:
        song_id: The ID of the song.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if like:
        return _get("spotify_like_song")(song_id=song_id)
    return _get("spotify_unlike_song")(song_id=song_id)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_toggle_album_like(album_id: int, like: bool) -> dict[str, Any]:
    """Like or unlike an album on Spotify.

    Args:
        album_id: The ID of the album.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if like:
        return _get("spotify_like_album")(album_id=album_id)
    return _get("spotify_unlike_album")(album_id=album_id)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_toggle_playlist_like(playlist_id: int, like: bool) -> dict[str, Any]:
    """Like or unlike a playlist on Spotify.

    Args:
        playlist_id: The ID of the playlist.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if like:
        return _get("spotify_like_playlist")(playlist_id=playlist_id)
    return _get("spotify_unlike_playlist")(playlist_id=playlist_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Follow/Unfollow artist
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_toggle_artist_follow(artist_id: int, follow: bool) -> dict[str, Any]:
    """Follow or unfollow an artist on Spotify.

    Args:
        artist_id: The ID of the artist.
        follow: True to follow, False to unfollow.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if follow:
        return _get("spotify_follow_artist")(artist_id=artist_id)
    return _get("spotify_unfollow_artist")(artist_id=artist_id)


# ---------------------------------------------------------------------------
# Symmetric pairs: Library add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_song_in_library(
    song_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a song from your Spotify library.

    Args:
        song_id: The ID of the song.
        action: "add" to add to library, "remove" to remove from library.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "add":
        return _get("spotify_add_song_to_library")(song_id=song_id)
    return _get("spotify_remove_song_from_library")(song_id=song_id)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_album_in_library(
    album_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove an album from your Spotify library.

    Args:
        album_id: The ID of the album.
        action: "add" to add to library, "remove" to remove from library.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "add":
        return _get("spotify_add_album_to_library")(album_id=album_id)
    return _get("spotify_remove_album_from_library")(album_id=album_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Playlist song add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_song_in_playlist(
    playlist_id: int,
    song_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a song from a Spotify playlist.

    Args:
        playlist_id: The ID of the playlist.
        song_id: The ID of the song.
        action: "add" to add the song, "remove" to remove the song.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "add":
        return _get("spotify_add_song_to_playlist")(
            playlist_id=playlist_id, song_id=song_id
        )
    return _get("spotify_remove_song_from_playlist")(
        playlist_id=playlist_id, song_id=song_id
    )


# ---------------------------------------------------------------------------
# Symmetric pair: Download/remove download
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage_song_download(
    song_id: int,
    action: Literal["download", "remove"],
) -> dict[str, Any]:
    """Download a song or remove it from downloads on Spotify.

    Args:
        song_id: The ID of the song.
        action: "download" to download, "remove" to remove from downloads.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if action == "download":
        return _get("spotify_download_song")(song_id=song_id)
    return _get("spotify_remove_downloaded_song")(song_id=song_id)


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "spotify_manage_playlist",
    "spotify_create_playlist",
    "spotify_show_playlist",
    "spotify_update_playlist",
    "spotify_delete_playlist",
)
mark_tools_absorbed_by(
    "spotify_manage_review",
    "spotify_show_song_reviews",
    "spotify_review_song",
    "spotify_update_song_review",
    "spotify_delete_song_review",
    "spotify_show_song_review",
    "spotify_show_album_reviews",
    "spotify_review_album",
    "spotify_update_album_review",
    "spotify_delete_album_review",
    "spotify_show_album_review",
    "spotify_show_playlist_reviews",
    "spotify_review_playlist",
    "spotify_update_playlist_review",
    "spotify_delete_playlist_review",
    "spotify_show_playlist_review",
)
mark_tools_absorbed_by(
    "spotify_toggle_song_like",
    "spotify_like_song",
    "spotify_unlike_song",
)
mark_tools_absorbed_by(
    "spotify_toggle_album_like",
    "spotify_like_album",
    "spotify_unlike_album",
)
mark_tools_absorbed_by(
    "spotify_toggle_playlist_like",
    "spotify_like_playlist",
    "spotify_unlike_playlist",
)
mark_tools_absorbed_by(
    "spotify_toggle_artist_follow",
    "spotify_follow_artist",
    "spotify_unfollow_artist",
)
mark_tools_absorbed_by(
    "spotify_manage_song_in_library",
    "spotify_add_song_to_library",
    "spotify_remove_song_from_library",
)
mark_tools_absorbed_by(
    "spotify_manage_album_in_library",
    "spotify_add_album_to_library",
    "spotify_remove_album_from_library",
)
mark_tools_absorbed_by(
    "spotify_manage_song_in_playlist",
    "spotify_add_song_to_playlist",
    "spotify_remove_song_from_playlist",
)
mark_tools_absorbed_by(
    "spotify_manage_song_download",
    "spotify_download_song",
    "spotify_remove_downloaded_song",
)
