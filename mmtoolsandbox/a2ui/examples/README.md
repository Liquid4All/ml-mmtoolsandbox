# A2UI Example Templates

UI templates for building interactive scenarios in MMToolSandbox. All templates use the **A2UI v0.8 stable** protocol format.

**Source:** [A2UI repository](https://github.com/google/A2UI) at commit `29f58e80f308d0c7efe8a1bcec7284d666dd9c6e` (Apr 3, 2026).

## Template Inventory

### Forms & Input

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `booking_form.json` | ADK | Multi-field form | TextField, DateTimeInput, Button, Image | Restaurant/hotel booking, appointment scheduling, reservation forms |
| `02_email-compose.json` | Spec | Email form | TextArea, TextField, Row, Button | Email drafting, message composition, contact form |
| `09_login-form.json` | Spec | Auth form | TextField, Button, Divider, Text | Login flow, signup form, authentication scenarios |
| `13_coffee-order.json` | Spec | Order form | Card, Column, Row, Button, Image | Food ordering, purchase cart, item customization |
| `19_software-purchase.json` | Spec | Purchase form | Card, Column, Row, Button, Text | Software licensing, subscription signup, payment flow |
| `restaurant-booking.json` | Theater | Booking workflow | Form + confirmation flow | End-to-end reservation, multi-step booking |

### Cards & Profiles

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `contact_card.json` | ADK | Detail card | Card, Image, Icon, Button, Text | Contact profile, employee details, user info display |
| `25_contact-card.json` | Spec | Contact info | Card, Row, Column, Text, Icon | Business card, directory entry, contact details |
| `05_product-card.json` | Spec | E-commerce card | Card, Image, Text, Row, Button | Product listing, shopping interface, item comparison |
| `08_user-profile.json` | Spec | Profile layout | Column, Image, Text, Row, Icon | User profile, team member card, account settings |
| `20_restaurant-card.json` | Spec | Restaurant detail | Card, Column, Row, Image, Text | Restaurant profile, venue details, place info |
| `24_recipe-card.json` | Spec | Recipe details | Card, Image, Column, Row, Text, Icon | Recipe display, cooking guide, ingredient breakdown |
| `29_movie-card.json` | Spec | Media card | Card, Image, Column, Text, Button | Movie listing, show recommendation, media details |
| `14_sports-player.json` | Spec | Player card | Card, Image, Row, Text, Icon | Player profile, athlete stats, team roster display |
| `22_credit-card.json` | Spec | Payment method | Card, Row, Text, Icon | Payment info display, card management, wallet view |

### Lists & Collections

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `contact_list.json` | ADK | Scrollable list | List (template-based), Card, Text | Contact search results, people finder, directory browse |
| `restaurant_list.json` | ADK | Item list with actions | List (template-based), Card, Image, Text, Button | Restaurant finder, product catalog, hotel/flight comparison |
| `two_column_list.json` | ADK | Two-column list | List, Row, Card, Image, Text | Restaurant comparison, hotel comparison, item pairing |
| `restaurant-grid.json` | Theater | Two-column grid | Grid layout, Card, Image, Text, Button | Side-by-side comparison, product grid, gallery view |
| `contact-lookup.json` | Theater | Search results | List, Card, Text, Button | Contact search, people finder, directory query |
| `18_track-list.json` | Spec | Playlist/tracklist | List, Card, Row, Image, Text, Button | Music library, playlist management, media collection |
| `03_calendar-day.json` | Spec | Calendar view | Grid-like layout, Text | Day planner, event scheduling, time slot booking |
| `12_chat-message.json` | Spec | Chat thread | Column, List, Row, Text | Messaging interface, chat history review, conversation display |

### Status & Info Cards

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `01_flight-status.json` | Spec | Status tracker | Card, Row, Divider, Text | Flight tracking, travel status, transit info display |
| `21_shipping-status.json` | Spec | Delivery tracker | Card, Column, Divider, Text | Package tracking, order status, milestone progress |
| `04_weather-current.json` | Spec | Weather display | Card, Icon, Image, Text, Row | Weather briefing, outdoor activity planning, travel conditions |
| `weather-widget.json` | Theater | Weather widget | Card, Icon, Text, Row | Weather display, location conditions, travel planning |
| `15_account-balance.json` | Spec | Financial card | Card, Row, Text, Icon | Bank balance display, account summary, budget overview |
| `16_workout-summary.json` | Spec | Activity stats | Card, Column, Row, Icon, Text | Fitness summary, health metrics, activity tracking |
| `27_stats-card.json` | Spec | KPI/metrics | Card, Text, Icon, Row | Analytics dashboard, performance metrics, data summary |
| `23_step-counter.json` | Spec | Progress metric | Card, Row, Icon, Text | Step goals, progress indicator, activity counter |
| `28_countdown-timer.json` | Spec | Timer display | Card, Text, Icon | Event countdown, deadline tracker, timer widget |

### Events & Activities

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `17_event-detail.json` | Spec | Event info | Card, Column, Row, Image, Text, Button | Event listing, conference info, RSVP flow |
| `07_task-card.json` | Spec | Task with checkbox | Card, CheckBox, Text, Button | To-do management, task tracking, checklist completion |
| `10_notification-permission.json` | Spec | Permission prompt | Card, Text, Button | Consent dialog, permission request, opt-in flow |

### Confirmations & Feedback

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `confirmation.json` | ADK | Confirmation dialog | Card, Divider, Button, Text | Order confirmation, action verification, booking review |
| `success_feedback.json` | ADK | Success state | Card, Icon, Text | Task completion, submission feedback, approval message |
| `11_purchase-complete.json` | Spec | Transaction receipt | Card, Column, Divider, Text, Icon | Purchase confirmation, invoice display, order receipt |

### Advanced & Multi-Surface

| Template | Source | UI Type | Components | Potential Scenarios |
|----------|--------|---------|------------|---------------------|
| `multi_surface.json` | ADK | Multi-panel layout | Multiple surfaces | Dashboard with side-by-side panels, split-screen workflows |
| `chart_node_click.json` | ADK | Interactive chart | Custom (node click) | Org charts, flowcharts, hierarchy navigation |
| `northstar-tour.json` | Theater | Guided tour | Multi-step layout | Onboarding flow, feature tour, step-by-step guide |

## Sources

| Label | Path in A2UI repo |
|-------|-------------------|
| **ADK** | `samples/agent/adk/{contact_lookup,restaurant_finder,custom-components-example}/examples/0.8/` |
| **Spec** | `specification/v0_8/json/catalogs/basic/examples/` |
| **Theater** | `tools/composer/src/data/theater/` |


## Usage

Templates are discoverable by the agent via `ui_get_item_details('Examples', '<name>')`. See [docs.py](../docs.py) for the documentation registry.

To use in a scenario, add the relevant tools to `tool_allow_list`:

```python
ScenarioExtension(
    ...,
    tool_allow_list=["ui_get_quick_start", "render_ui_screen", "show_ui_to_user", ...],
    user_tool_allow_list=["ui_user_interact"],
)
```
