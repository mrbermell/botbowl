# 2020 ruleset feature list

The purpose of this document gather a summary of the delta between the 2016 and 2020 rules, and showing the progress 
towards an implementation of the 2020 ruleset. The text below might not be perfect, check the rulebook before coding!    


## Terminology 
The 2020 rules coins a few new terms which are not rules but helps explain the rules.

| Term          | Meaning  |
|---------------|----------|
| Open player   | Player not in any opponent's tacklezone   |
| Marked player | Player in one or more of opponent's tacklezones  |
| Rush          | Rename of GFI |
| Deviate       | Moving the ball D6 squares in direction determined by a D8 (e.g. kickoff)   |
| Scatter       | Move ball three squares in direction determined by three subsequent D8 (e.g. inaccurate pass)|
| Bounce        | Move ball one square in direction determined by D8 (e.g. failed catch/pickup)  |
| Stalling      | Player can score with full certainty but chooses not to |



## Prayers to Nuffle
For every 50k in Team Value difference the underdog coach gets one roll (D16) on the Prayers to Nuffle table. This in addition to Petty Cash. This is also a kickoff result.   

| Prayer                        | Implemented   | Has Test(s) | 
|-------------------------------|---------------|-----------|
| 1 - Treacherou Trapdoor       |             |      |
| 2 - Friends with the Ref      |             |      |
| 3 - Stiletto                  |             |      |
| 4 - Iron Man                  |             |      |
| 5 - Knuckle Duster            |             |      |
| 6 - Bad Habits                |             |      |
| 7 - Greasy Cleats             |             |      |
| 8 - Blessed Statue of Nuffle  |             |      |
| 9 - Moles under the Pitch     |             |      |
| 10 - Perfect Passing          |             |      |
| 11 - Fan Interaction          |             |      |
| 12 - Necessary Violence       |             |      |
| 13 - Fouling Frenzy           |             |      |
| 14 - Throw a Rock             |             |      |
| 15 - Under Scrutiny           |             |      |
| 16 - Intensive Training       |             |      |




## Kickoff event table 

Fan Factor is used instead of Fame in some places. 

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-----------|-------|
| 2 - Get the ref        | ✔️         | ❌        | Same |
| 3 - Time out           |           |        | (old Riot) Kicking team's turn marker on 6,7,8 then substract one turn, else add one.  |
| 4 - Solid defence      |           |        | (old Perfect Defence) Limited to D3+3 players  |
| 5 - High Kick          | ✔         | ✔        | Same |
| 6 - Cheering Fans      |           |        | D6+fan factor. Winner rolls a Player to Nuffle|
| 7 - Brilliant Coaching | ✔         | ✔        | Same |
| 8 - Changing Weather   | ✔         | ?        | If the new weather is Perfect Conditions the ball now Scatters (i.e. 3D8) |
| 9 - Quicksnap          |           |        | Limited to D3+3 open players|
| 10 - Blitz             |           |        | D3+3 open players  |
| 11 - Officious Ref     |           |        | D6+fan factor to determine coach. Randomly select player. D6: 1: Sent to dungeon. 2+ Stunned |
| 12 - Pitch invasion!   |           |        | D6+fan factor to determine coach. Randomly select D3 players. Stunned  |



## Skills 
'Jump' refers to the action of Jumping over Prone players which all players can do now.  

### Removed skills 
Skills that are completely removed are: Piling On, 

### Agility 
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Catch              | ✔         |           | same |
|Diving Catch       | ✔         |           | same |
|Diving Tackle      |           |           | works on leap and jump too |
|Dodge              | ✔         |           | same |
|Defensive          |           |           | new: cancels adjescent opponent's Guard during Opponent's turn. |
|Jump Up            | ✔         |           | same |
|Leap               |           |           | may Jump over any type of square, negative modifer reduce by 1, to minimum of -1 |
|Safe Pair of Hands |           |           | new |
|Sidestep           | ✔         |           | same |
|Sneaky Git         |           |           | not sent of because of doubles on armor even if it breaks, may move after the foul |
|Sprint             | ✔         |           | same |
|Sure Feet          | ✔         |           | same |

### General
| Skill            | Implemented | Has Test(s) | Change  |
|------------------|-------------|-------------|-------|
|Block             | ✔         | ✔          | same |
|Dauntless         | ✔         |            | same |
|Dirty player (+1) | ✔         |            | same |
|Fend              | ✔         |            | same |
|Frenzy            | ✔         | ✔          | same |
|Kick              |           |            | same |
|Pro               |           |            | +3 instead of +4. May only re-roll one dice |
|Shadowing         |           |            | success determined by D6 +own MA -opp MA > 5  |
|Strip Ball        | ✔         | ✔          | same |
|Sure Hands        | ✔         | ✔          | same |
|Tackle            | ✔         |            | same |
|Wrestle           | ✔         |            | same |

### Mutations
| Skill              | Implemented | Has Test(s) | Change  |
|--------------------|------------|-------------|-------|
|Big Hand            | ✔        | ✔         | same |
|Claws               |          |           | doesn't stack with Mightly blow |
|Disturbing Presence | ✔        |           | same |
|Extra Arms          | ✔        | ✔         | same |
|Foul Appearance     | ✔        |           | same |
|Horns               | ✔        |           | same |
|Iron Hard Skin      |          |           | new: Immune to Claw |
|Monstrous Mouth     |          |           | new: Catch re-roll and immune to Strip Ball |
|Prehensile Tail     |          |           | works on Leap and Jump |
|Tentacles           |          |           | success determined as D6 +own St -opp ST > 5 |
|Two Heads           | ✔        |           | same |
|Very Long Legs      |          |           | negative modifers for Jump and Leap (if player has skill) reduce by 1, to minimum of -1, Immune to Cloud Burster |

### Passing
| Skill          | Implemented | Has Test(s) | Change  |
|----------------|-------------|-------------|-------|
|Accurate        |              |           | Only quick pass and short pass |
|Connoneer       |              |           | as Accurate but on Long pass and Long Bomb |
|Cloud Burster   |              |           | New: choose if opposing coach shall re-roll a successful Interfere when throwing Long or Long Bomb  |
|Dump-Off        | ✔            |           | same |
|Fumblerooskie   |              |           | new: leave ball in vacated square during movement.  dice involved |
|Hail Mary Pass  |              |           | tacklezones matter |
|Leader          |              |           | same |
|Nerves of Steel | ✔            | ✔         | same |
|On the Ball     |              |           | Kick off return and pass block combined |
|Pass            | ✔            | ✔         | same |
|Running Pass    |              |           | new: may continue moving after quick pass |
|Safe Pass       |              |           | Fumbled passes doesn't cause bounce nor turnover |


### Strength
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Arm Bar            |           |           | new |
|Brawler            |           |           | new |
|Break Tackle       |           |           | +2 on dodge if ST>4 else +1 once per turn  |
|Grab               | ✔         |           | same |
|Guard              |           |           | works on fouls too |
|Juggernaut         | ✔         |           | same |
|Mighty Blow +1     |           |           | doesn't work passively (e.g. attacker down), +X |
|Multiple Block     |           |           | same |
|Pile Driver        |           |           | As piling on but is evaluate as a foul |
|Stand Firm         | ✔         |           | same |
|Strong Arm         |           |           | Only applicable for Throw Team-mate |
|Thick Skull        | ✔         |           | same |

### Traits
| Skill             | Implemented | Has Test(s) | Change  |
|-------------------|-------------|-------------|-------|
|Animal Savagery    |           |           | new |
|Animosity          |           |           | same? |
|Always Hungry      | ✔         | ✔         | same |
|Ball & Chain       |           |           | ? |
|Bombardier         |           |           | ? |
|Bone Head          | ✔         | ✔         | same |
|Chainsaw           |           |           | ? |
|Decay              |           |           | ? |
|Hypnotic Gaze      | ✔         | ✔         | same |
|Kick Team-mate     |           |           | ? |
|Loner (+X)         |           |           | +X is new |
|No Hands           | ✔         |           | same |
|Plague Ridden      |           |           | new |
|Pogo Stick         |           |           | ? |
|Projectile Vomit   |           |           | new  |
|Really Stupid      | ✔         | ✔         | same |
|Regeneration       | ✔         | ✔         | same |
|Right Stuff        | ✔         | ✔         | same? |
|Secret Weapon      |           |           | same |
|Stab               | ✔         |           | same |
|Stunty             |           |           |  modifier for passing, but opponent gets +1 when interfering with pass from Stunty Player |
|Swarming           |           |           | new |
|Swoop              |           |           | ? |
|Take Root          | ✔         | ✔         | same |
|Titchy             |           |           | Same but also can never cause negative modifiers to opponent player's agility test (e.g. catching/throwing ball) |
|Throw Team-mate    |           |           | ? |
|Timmm-ber!         | ✔         | ✔         | same |
|Unchannelled Fury  |           |           | new |


## Passing

| Feature                | Implemented | Has Test(s) | Change  |
|------------------------|-------------|-------------|-------|
| Passing characteristic |           |           | AG does  longer determine passing ability |
| New distance modifiers |           |           | 0 for quick, ... , -3 for Long Bombs |
| Wildly inaccurate pass |           |           | Roll is 1 after modifiers. Deviates (like a kickoff) |
| Catch modifiers        |           |           | 0 for accurate |
| Pass Interference      |           |           | Replaces old interception rules |


## Other feature changes
Fame is removed and replaced with a similar 'Fan Factor' 


| Feature                 | Implemented | Has Test(s) | Change  |
|-------------------------|-------------|-------------|---------|
| Fan factor              |           |           | D3 + nbr of Dedicated Fans the team has |
| Team re-rolls           |           |           | Multiple team re-rolls can be used per turn |
| Player characteristic   |           |           | AG and AV is now e.g. 3+ instead of 4 |
| Passing characteristic  |           |           | new |
| Sweltering heat         |           |           | D3 players, randomly selected |
| Jump over prone players |           |           | New  |
| Niggling Injury         |           |           | +1 on Casualty roll instead |
| Casualty table          |           |           | D16 table  |
| Stunty Injury table     |           |           | 2-6 stunned, 7-8 KO, 9 Badly Hurt (without casualty roll), 10+ Casualty  |



## Races 

All races have changed. This table shows what is working. 

| Race             | Missing positions       | Missing skills                     | Have icons    |
|------------------|-------------------------|------------------------------------|----------------
| Amazon             |                         |                                    | ✔    |
| Black Orc          |                         |                                    |    |
| Chaos Dwarf        |                         |                                    | ✔    |
| Chaos Choosen      |                         |                                    | ✔    |
| Chaos Renegades    |                         |                                    | Maybe? |
| Dark Elf           |                         |                                    | ✔    |
| Dwarf              |                         |                                    |      |
| Elven Union        |                         |                                    | ✔    |
| Goblin             |                         |                                    | Some?  |
| Halfling           |                         |                                    | ✔    |
| High Elf           |                         |                                    | ✔    |
| Human**            |                         |                                    | ✔    |
| Imperial Nobility  |                         |                                    |      |
| Tomb Kings         |                         |                                    | ✔    |
| Lizardmen          |                         |                                    | ✔    |
| Necromantic        |                         |                                    |      |
| Norse              |                         |                                    | ✔    |
| Nurgle*            |                         |                                    |      |
| Ogre               |                         |                                    | ✔    |
| Old World Alliance |                         |                                    |     |
| Orc                |                         |                                    | ✔    |
| Shambling Undead   |                         |                                    | ✔    |
| Skaven             |                         |                                    | ✔    |
| Snotling           |                         |                                    |     |
| Vampire            |                         |                                    | ✔    |
| Underworld Denizens|                         |                                    | Maybe? |         
| Wood Elf           |                         |                                    | ✔   | 

* Nurgle's Rot needs to be implemented in the post-game sequence
