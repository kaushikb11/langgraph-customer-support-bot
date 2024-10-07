# Swiss Airlines Customer Support Assistant

This project implements an intelligent customer support assistant for Swiss Airlines using a combination of language models and a graph-based dialog management system.

## Overview

The Swiss Airlines Customer Support Assistant is designed to handle various customer queries and tasks related to flight bookings, car rentals, hotel reservations, and trip recommendations. It uses a sophisticated dialog management system to route user requests to specialized assistants for specific tasks.

## Key Features

1. **Multi-task Handling**: The assistant can handle various tasks including:

   - Flight information and updates
   - Car rental bookings
   - Hotel reservations
   - Trip recommendations and excursions

1. **Specialized Assistants**: The system uses dedicated assistants for each task type, allowing for more focused and efficient handling of user requests.

1. **Dynamic Dialog Management**: The dialog flow is managed using a state graph, allowing for flexible routing between different assistants based on user needs.

1. **Database Integration**: The assistant interacts with a SQLite database to fetch and update information about flights, bookings, and other travel-related data.

1. **Safety Measures**: The system implements a two-tier tool system (safe and sensitive) to ensure proper handling of critical operations.

1. **User Approval for Sensitive Actions**: Before executing sensitive operations, the system requests user approval, adding an extra layer of security.
