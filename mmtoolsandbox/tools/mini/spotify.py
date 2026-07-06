# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI Spotify tools -- 3 workflow-based tools covering all Spotify functionality."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.spotify as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Tool 1: spotify_player -- "I want to control music playback"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_player(
    action: Literal[
        "play",
        "pause",
        "next",
        "previous",
        "seek",
        "loop",
        "shuffle",
        "show_queue",
        "add_to_queue",
        "remove_from_queue",
        "clear_queue",
        "move_in_queue",
        "show_volume",
        "set_volume",
        "show_current",
    ],
    # play params
    song_id: int | None = None,
    album_id: int | None = None,
    playlist_id: int | None = None,
    queue_position: int | None = None,
    # seek param
    seek_seconds: int | NotGiven = NOT_GIVEN,
    # loop param
    loop: bool | NotGiven = NOT_GIVEN,
    # queue params
    position: int | NotGiven = NOT_GIVEN,
    current_position: int | NotGiven = NOT_GIVEN,
    new_position: int | NotGiven = NOT_GIVEN,
    # volume param
    volume: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Control Spotify music playback, queue, and volume.

    Actions and required params:
        play: Resume or play a song/album/playlist. Optionally pass one of
            song_id, album_id, playlist_id, or queue_position.
        pause: Pause the current song.
        next: Skip to next song.
        previous: Go to previous song.
        seek: Seek to a position. Requires seek_seconds.
        loop: Toggle loop on current song. Requires loop (bool).
        shuffle: Shuffle the song queue.
        show_queue: Show the current song queue.
        add_to_queue: Add song/album/playlist to queue. Pass one of
            song_id, album_id, or playlist_id.
        remove_from_queue: Remove song at position. Requires position.
        clear_queue: Clear the entire song queue.
        move_in_queue: Move a song. Requires current_position, new_position.
        show_volume: Show current volume level.
        set_volume: Set volume. Requires volume (int).
        show_current: Show the currently playing song.

    Args:
        action: The playback action to perform.
        song_id: Song ID (for play, add_to_queue).
        album_id: Album ID (for play, add_to_queue).
        playlist_id: Playlist ID (for play, add_to_queue).
        queue_position: Queue position to play from (for play).
        seek_seconds: Position in seconds (for seek).
        loop: Whether to loop the current song (for loop).
        position: Queue position (for remove_from_queue).
        current_position: Current queue position (for move_in_queue).
        new_position: New queue position (for move_in_queue).
        volume: Volume level 0-100 (for set_volume).

    Returns:
        For play/pause/next/previous/seek/loop/shuffle: current playback
            status dict with song_id, title, artist, position.
        For show_queue: list of song dicts in queue order.
        For add_to_queue/remove_from_queue/clear_queue/move_in_queue:
            updated queue confirmation.
        For show_volume/set_volume: dict with volume level.
        For show_current: current song dict.
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
    elif action == "show_queue":
        return _get("spotify_show_song_queue")()
    elif action == "add_to_queue":
        kwargs = {}
        if song_id is not None:
            kwargs["song_id"] = song_id
        if album_id is not None:
            kwargs["album_id"] = album_id
        if playlist_id is not None:
            kwargs["playlist_id"] = playlist_id
        return _get("spotify_add_to_queue")(**kwargs)
    elif action == "remove_from_queue":
        return _get("spotify_remove_song_from_queue")(position=position)
    elif action == "clear_queue":
        return _get("spotify_clear_song_queue")()
    elif action == "move_in_queue":
        return _get("spotify_move_song_in_queue")(
            current_position=current_position, new_position=new_position
        )
    elif action == "show_volume":
        return _get("spotify_show_volume")()
    elif action == "set_volume":
        return _get("spotify_set_volume")(volume=volume)
    elif action == "show_current":
        return _get("spotify_show_current_song")()
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Tool 2: spotify_discover -- "I want to find or browse music"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_discover(
    domain: Literal[
        "search",
        "show",
        "liked",
        "library",
        "privates",
        "genres",
        "recommendations",
        "downloaded",
        "following",
    ],
    # search/show/liked/library/privates need entity_type
    entity_type: Literal["song", "album", "playlist", "artist"] | NotGiven = NOT_GIVEN,
    entity_id: int | NotGiven = NOT_GIVEN,
    # search filters (all None = not passed)
    query: str | None = "",
    genre: str | None = None,
    artist_id: int | None = None,
    album_id: int | None = None,
    min_release_date: str | None = None,
    max_release_date: str | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
    min_like_count: int | None = None,
    max_like_count: int | None = None,
    min_play_count: int | None = None,
    max_play_count: int | None = None,
    owner_email: str | None = None,
    min_follower_count: int | None = None,
    max_follower_count: int | None = None,
    # pagination
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Find or browse music on Spotify: search, view details, or list collections.

    Domains and required params:
        search: Search songs/albums/playlists/artists. Requires entity_type.
            Supports filters: query, genre, artist_id, album_id,
            min/max_release_date, min/max_duration, min/max_rating,
            min/max_like_count, min/max_play_count, owner_email,
            min/max_follower_count, page_index, page_limit, sort_by.
        show: View details of a single entity. Requires entity_type and
            entity_id. Supports song, album, artist.
        liked: List your liked songs/albums/playlists. Requires entity_type.
            Supports page_index, page_limit, sort_by.
        library: List songs/albums in your library. entity_type must be
            "song" or "album". Supports query, page_index, page_limit,
            sort_by.
        privates: Show private info for an entity. Requires entity_type
            (song/album/playlist) and entity_id.
        genres: List all available genres. No extra params needed.
        recommendations: Get personalized song recommendations.
            Supports page_index, page_limit.
        downloaded: List your downloaded songs. No extra params needed.
        following: List artists you follow. If entity_id is provided,
            checks if you follow that specific artist.

    Args:
        domain: The browse domain to interact with.
        entity_type: Type of entity (song, album, playlist, artist).
        entity_id: Entity ID (for show, privates, following check).
        query: Search query string (for search, library).
        genre: Genre filter (for search).
        artist_id: Artist ID filter (for search).
        album_id: Album ID filter (for search).
        min_release_date: Min release date filter (for search).
        max_release_date: Max release date filter (for search).
        min_duration: Min duration in seconds (for search).
        max_duration: Max duration in seconds (for search).
        min_rating: Min rating filter (for search).
        max_rating: Max rating filter (for search).
        min_like_count: Min like count filter (for search).
        max_like_count: Max like count filter (for search).
        min_play_count: Min play count filter (for search).
        max_play_count: Max play count filter (for search).
        owner_email: Owner email filter (for search playlists).
        min_follower_count: Min follower count (for search artists).
        max_follower_count: Max follower count (for search artists).
        page_index: Page index for pagination.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix.

    Returns:
        For search: list of entity dicts with entity-specific IDs
            (song_id, album_id, playlist_id, or artist_id).
        For show: single entity detail dict.
        For liked/library/following: list of entity dicts.
        For genres: list of genre strings.
        For recommendations: list of song dicts.
        For downloaded: list of downloaded song dicts.
    """
    if domain == "search":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for search")
        func_name = f"spotify_search_{entity_type}s"
        kwargs: dict[str, Any] = {}
        # Common filters
        if query is not None:
            kwargs["query"] = query
        if genre is not None:
            kwargs["genre"] = genre
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        # Song-specific filters
        if entity_type == "song":
            if artist_id is not None:
                kwargs["artist_id"] = artist_id
            if album_id is not None:
                kwargs["album_id"] = album_id
            if min_release_date is not None:
                kwargs["min_release_date"] = min_release_date
            if max_release_date is not None:
                kwargs["max_release_date"] = max_release_date
            if min_duration is not None:
                kwargs["min_duration"] = min_duration
            if max_duration is not None:
                kwargs["max_duration"] = max_duration
            if min_rating is not None:
                kwargs["min_rating"] = min_rating
            if max_rating is not None:
                kwargs["max_rating"] = max_rating
            if min_like_count is not None:
                kwargs["min_like_count"] = min_like_count
            if max_like_count is not None:
                kwargs["max_like_count"] = max_like_count
            if min_play_count is not None:
                kwargs["min_play_count"] = min_play_count
            if max_play_count is not None:
                kwargs["max_play_count"] = max_play_count
        # Album-specific filters
        elif entity_type == "album":
            if min_rating is not None:
                kwargs["min_rating"] = min_rating
            if max_rating is not None:
                kwargs["max_rating"] = max_rating
            if min_release_date is not None:
                kwargs["min_release_date"] = min_release_date
            if max_release_date is not None:
                kwargs["max_release_date"] = max_release_date
            if min_like_count is not None:
                kwargs["min_like_count"] = min_like_count
            if max_like_count is not None:
                kwargs["max_like_count"] = max_like_count
        # Playlist-specific filters
        elif entity_type == "playlist":
            if min_like_count is not None:
                kwargs["min_like_count"] = min_like_count
            if max_like_count is not None:
                kwargs["max_like_count"] = max_like_count
            if min_rating is not None:
                kwargs["min_rating"] = min_rating
            if max_rating is not None:
                kwargs["max_rating"] = max_rating
            if owner_email is not None:
                kwargs["owner_email"] = owner_email
        # Artist-specific filters
        elif entity_type == "artist":
            if min_follower_count is not None:
                kwargs["min_follower_count"] = min_follower_count
            if max_follower_count is not None:
                kwargs["max_follower_count"] = max_follower_count
        return _get(func_name)(**kwargs)

    elif domain == "show":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for show")
        id_param = f"{entity_type}_id"
        return _get(f"spotify_show_{entity_type}")(**{id_param: entity_id})

    elif domain == "liked":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for liked")
        kwargs = {}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get(f"spotify_show_liked_{entity_type}s")(**kwargs)

    elif domain == "library":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for library")
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get(f"spotify_show_{entity_type}_library")(**kwargs)

    elif domain == "privates":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for privates")
        id_param = f"{entity_type}_id"
        return _get(f"spotify_show_{entity_type}_privates")(**{id_param: entity_id})

    elif domain == "genres":
        return _get("spotify_show_genres")()

    elif domain == "recommendations":
        kwargs = {}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("spotify_show_recommendations")(**kwargs)

    elif domain == "downloaded":
        return _get("spotify_show_downloaded_songs")()

    elif domain == "following":
        if entity_id is not NOT_GIVEN:
            return _get("spotify_show_artist_following")(artist_id=entity_id)
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("spotify_show_following_artists")(**kwargs)

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Tool 3: spotify_manage -- "I want to manage my music collection"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def spotify_manage(
    domain: Literal[
        "playlist", "review", "like", "follow", "library", "playlist_song", "download"
    ],
    action: Literal[
        "create", "show", "update", "delete", "list", "add", "remove", "toggle", "write"
    ]
    | NotGiven = NOT_GIVEN,
    entity_type: Literal["song", "album", "playlist"] | NotGiven = NOT_GIVEN,
    entity_id: int | NotGiven = NOT_GIVEN,
    playlist_id: int | NotGiven = NOT_GIVEN,
    song_id: int | NotGiven = NOT_GIVEN,
    like: bool | NotGiven = NOT_GIVEN,
    follow: bool | NotGiven = NOT_GIVEN,
    # playlist params
    title: str | NotGiven = NOT_GIVEN,
    is_public: bool | NotGiven = NOT_GIVEN,
    # review params
    rating: int | NotGiven = NOT_GIVEN,
    review_title: str | NotGiven = NOT_GIVEN,
    text: str | NotGiven = NOT_GIVEN,
    review_id: int | NotGiven = NOT_GIVEN,
    # pagination
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage your Spotify music collection: playlists, reviews, likes, follows,
    library, playlist songs, and downloads.

    Domains and actions:
        playlist: CRUD for playlists.
            create: Requires title. is_public defaults to false.
            show: Requires entity_id (the playlist ID).
            update: Requires entity_id, plus title and/or is_public.
            delete: Requires entity_id.
            list: List your playlists. Supports page_index, page_limit,
                sort_by.
        review: Manage reviews for songs/albums/playlists.
            Requires entity_type.
            list: List reviews. Requires entity_id.
            show: Show a review. Requires review_id.
            write: Write a review. Requires entity_id, rating.
                Optional: review_title, text.
            update: Update a review. Requires review_id. Optional:
                rating, review_title, text.
            delete: Delete a review. Requires review_id.
        like: Like or unlike a song/album/playlist.
            Requires entity_type, entity_id, like (bool).
        follow: Follow or unfollow an artist.
            Requires entity_id, follow (bool).
        library: Add/remove songs or albums from your library.
            Requires entity_type (song/album), entity_id, action
            (add/remove).
        playlist_song: Add/remove a song in a playlist.
            Requires playlist_id, song_id, action (add/remove).
        download: Download or remove a downloaded song.
            Requires entity_id, action (add/remove).

    Args:
        domain: The collection domain to manage.
        action: The specific action to perform.
        entity_type: Type of entity (song, album, playlist).
        entity_id: Entity ID.
        playlist_id: Playlist ID (for playlist_song).
        song_id: Song ID (for playlist_song).
        like: Whether to like (true) or unlike (false).
        follow: Whether to follow (true) or unfollow (false).
        title: Playlist title (for playlist create/update).
        is_public: Playlist visibility (for playlist create/update).
        rating: Review rating (for review write/update).
        review_title: Review title (for review write/update).
        text: Review text (for review write/update).
        review_id: Review ID (for review show/update/delete).
        page_index: Page index for pagination.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix.

    Returns:
        For playlist create: dict with playlist_id. Pass to playlist
            show/update/delete and playlist_song actions.
        For playlist delete: confirmation. Irreversible.
        For review write: dict with review_id.
        For review delete: confirmation. Irreversible.
        For like/follow/library/download: confirmation dict.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Spotify.
    """
    if domain == "playlist":
        if action == "create":
            kwargs: dict[str, Any] = {"title": title}
            if is_public is not NOT_GIVEN:
                kwargs["is_public"] = is_public
            return _get("spotify_create_playlist")(**kwargs)
        elif action == "show":
            return _get("spotify_show_playlist")(playlist_id=entity_id)
        elif action == "update":
            kwargs = {"playlist_id": entity_id}
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            if is_public is not NOT_GIVEN:
                kwargs["is_public"] = is_public
            return _get("spotify_update_playlist")(**kwargs)
        elif action == "delete":
            return _get("spotify_delete_playlist")(playlist_id=entity_id)
        elif action == "list":
            kwargs = {}
            if page_index is not None:
                kwargs["page_index"] = page_index
            if page_limit is not None:
                kwargs["page_limit"] = page_limit
            if sort_by is not None:
                kwargs["sort_by"] = sort_by
            return _get("spotify_show_playlist_library")(**kwargs)
        else:
            raise ValueError(f"Unknown playlist action: {action}")

    elif domain == "review":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for review domain")
        id_param = f"{entity_type}_id"

        if action == "list":
            kwargs = {id_param: entity_id}
            if page_index is not None:
                kwargs["page_index"] = page_index
            if page_limit is not None:
                kwargs["page_limit"] = page_limit
            if sort_by is not None:
                kwargs["sort_by"] = sort_by
            return _get(f"spotify_show_{entity_type}_reviews")(**kwargs)
        elif action == "show":
            return _get(f"spotify_show_{entity_type}_review")(review_id=review_id)
        elif action == "write":
            kwargs = {id_param: entity_id, "rating": rating}
            if review_title is not NOT_GIVEN:
                kwargs["title"] = review_title
            if text is not NOT_GIVEN:
                kwargs["text"] = text
            return _get(f"spotify_review_{entity_type}")(**kwargs)
        elif action == "update":
            kwargs = {"review_id": review_id}
            if rating is not NOT_GIVEN:
                kwargs["rating"] = rating
            if review_title is not NOT_GIVEN:
                kwargs["title"] = review_title
            if text is not NOT_GIVEN:
                kwargs["text"] = text
            return _get(f"spotify_update_{entity_type}_review")(**kwargs)
        elif action == "delete":
            return _get(f"spotify_delete_{entity_type}_review")(review_id=review_id)
        else:
            raise ValueError(f"Unknown review action: {action}")

    elif domain == "like":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for like domain")
        id_param = f"{entity_type}_id"
        if like is NOT_GIVEN:
            raise ValueError("like (bool) is required for like domain")
        if like:
            return _get(f"spotify_like_{entity_type}")(**{id_param: entity_id})
        else:
            return _get(f"spotify_unlike_{entity_type}")(**{id_param: entity_id})

    elif domain == "follow":
        if follow is NOT_GIVEN:
            raise ValueError("follow (bool) is required for follow domain")
        if follow:
            return _get("spotify_follow_artist")(artist_id=entity_id)
        else:
            return _get("spotify_unfollow_artist")(artist_id=entity_id)

    elif domain == "library":
        if entity_type is NOT_GIVEN:
            raise ValueError("entity_type is required for library domain")
        id_param = f"{entity_type}_id"
        if action == "add":
            return _get(f"spotify_add_{entity_type}_to_library")(
                **{id_param: entity_id}
            )
        elif action == "remove":
            return _get(f"spotify_remove_{entity_type}_from_library")(
                **{id_param: entity_id}
            )
        else:
            raise ValueError(f"Unknown library action: {action}")

    elif domain == "playlist_song":
        if action == "add":
            return _get("spotify_add_song_to_playlist")(
                playlist_id=playlist_id, song_id=song_id
            )
        elif action == "remove":
            return _get("spotify_remove_song_from_playlist")(
                playlist_id=playlist_id, song_id=song_id
            )
        else:
            raise ValueError(f"Unknown playlist_song action: {action}")

    elif domain == "download":
        if action == "add":
            return _get("spotify_download_song")(song_id=entity_id)
        elif action == "remove":
            return _get("spotify_remove_downloaded_song")(song_id=entity_id)
        else:
            raise ValueError(f"Unknown download action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
